"""Pick the SAM 3 text prompts for a spatial question about a video.

    >>> main("If I am standing by the stove and facing the sofa, is the tv to my front-left?")
    ['stove', 'sofa', 'tv']

Every noun in the question is a candidate. spaCy's parse reduces each one to its head noun plus
the modifiers that describe what the thing looks like: compound nouns and adjectives ("kitchen
table", "red chair"), without articles, numbers, possessives, superlatives ("longest"), ordinals
("first") or spatial words ("left", "front"). Heads are lemmatised ("chairs" -> "chair").

WordNet then decides whether the candidate is something physical that SAM 3 could segment:

  physical  a sense in PHYSICAL_LEXNAMES (artifacts, natural objects, people, animals, plants,
            food, body parts, substances), a top-level sense below physical_entity ("person",
            "animal"), or, as the word's main sense, a group whose members are physical
            ("people", "herd")

A word with several meanings is kept when any of them is physical ("play it safe"), with two
exceptions, because WordNet has rare physical senses for most of the vocabulary that spatial
questions are written in ("length" a length of timber, "size" glue, "meter" an instrument,
"axis" a vertebra, "back" the body part):

  - its main sense is an attribute, quantity, place, shape, time, relation, idea or act
    (QUESTION_LEXNAMES) and its physical senses are rare: under MIN_PHYSICAL_SHARE of its sense
    counts in SemCor (WordNet's sense-tagged corpus; a word missing from SemCor counts each
    sense once). Drops "length", "size", "side", "point";
    keeps "seat" and "post", whose main WordNet sense is a place but whose physical senses are
    common, and "table", "tv", "sign", "light", whose main senses are a group, a broadcast, a
    message, a phenomenon
  - it is mostly a unit of measurement (MIN_UNIT_SHARE of its sense counts): "meter", "foot"

A multi-word name that is itself in WordNet decides first when it is physical or a named place
("parking meter", "bus stop", "parking lot"), or when it is measurement vocabulary ("square
meter", "Cartesian plane", "y-axis"). When the head is dropped, its compound modifiers are tried
on their own: "the table height" -> "table". Two adjacent nouns that WordNet names together are
joined even when the parser split them ("closer to the trash | can").

Some words pass the WordNet test but are not useful prompts, and are dropped by word lists, which
apply to any question: spatial-relation words (SPATIAL_WORDS: "back", "corner"), generic nouns
(GENERIC_NOUNS: "object", "item", "pair"), and words for the video or its viewer (VIEWPOINT_NOUNS:
"video", "camera", "frame"). The last two only apply to a bare noun: "door frame" and "video
camera" are kept. The nominal the questioner says they are is dropped too ("You are a robot ...").
Words WordNet does not know are kept.

Syntactic mode (--syntactic, main(question, syntactic=True)) stops after the parse: every noun
phrase becomes a prompt, reduced to its head noun (spaCy's lemma) and all its compound and
adjectival modifiers. Nothing that depends on meaning is applied: no WordNet, no word lists, no
superlative filtering, no "You are a robot" rule. Only the grammar steps stay: the clean-up of the
question text, and keeping a modifier only when no comma, conjunction or article separates it
from its noun. "the length of the longest dimension of the stove" -> "length", "longest
dimension", "stove". Useful as a baseline, and it does not need the WordNet data.

Usage (with spacy and nltk installed, see below):
  python scene_sheet_creation/prompt_object_extractor.py "How far is the red car from the stove?"
  python scene_sheet_creation/prompt_object_extractor.py --explain "What is the size of this room?"
  python scene_sheet_creation/prompt_object_extractor.py --file questions.txt
  python scene_sheet_creation/prompt_object_extractor.py --syntactic --explain "..."

  Prints one line per question with its prompts joined by commas, the form scene_sheet_creator.py
  takes with --prompts. --explain also lists every candidate with the reason it was kept or
  dropped.

  From Python: main(question) -> list of prompts; analyse(question) -> every candidate with its
  verdict. Both take syntactic=True for syntactic mode.

Requirements:
  uv pip install spacy nltk          (or pip install)
  python -m spacy download en_core_web_md        needs pip in the venv; in a uv venv instead:
  uv pip install "en_core_web_md @ https://github.com/explosion/spacy-models/releases/download/en_core_web_md-3.8.0/en_core_web_md-3.8.0-py3-none-any.whl"
  WordNet is downloaded through nltk on first use if it is missing. en_core_web_md is the default
  because en_core_web_sm mis-tags rarer nouns ("fire hydrant"); pass --model to change it.
"""

import argparse
import re
import sys
from dataclasses import dataclass
from functools import cache
from itertools import pairwise

import nltk
from nltk.corpus import wordnet as wn

SPACY_MODEL = "en_core_web_md"  # en_core_web_sm tags rarer nouns ("hydrant") as adverbs

# WordNet lexicographer files ("lexnames") of physical things
PHYSICAL_LEXNAMES = {
    "noun.artifact",
    "noun.object",
    "noun.person",
    "noun.animal",
    "noun.plant",
    "noun.food",
    "noun.body",
    "noun.substance",
}
# lexnames of the vocabulary questions are written in rather than of things in the scene. A word
# whose main sense is one of these is dropped unless its physical senses are common. noun.Tops
# only matters for its abstract members ("space"): the physical ones ("person") are kept anyway.
QUESTION_LEXNAMES = {
    "noun.attribute",
    "noun.quantity",
    "noun.shape",
    "noun.location",
    "noun.time",
    "noun.relation",
    "noun.cognition",
    "noun.act",
    "noun.Tops",
}
MIN_PHYSICAL_SHARE = (
    0.3  # seat 0.44, post 0.45, corner 0.32 | center 0.2, side 0.18, meter 0.14
)
MIN_UNIT_SHARE = 1 / 3  # foot 0.43, yard 0.72, meter 0.86 | cup, box, hand 0

# spatial relations and directions: what the question asks about, not what to segment. Dropped as
# nouns ("to my back-left", "the corner") and as modifiers ("the front door" -> "door")
SPATIAL_WORDS = {
    "left",
    "right",
    "front",
    "back",
    "rear",
    "behind",
    "ahead",
    "top",
    "bottom",
    "side",
    "middle",
    "center",
    "centre",
    "corner",
    "edge",
    "end",
    "inside",
    "outside",
    "interior",
    "exterior",
    "near",
    "far",
    "nearby",
    "close",
    "distant",
    "adjacent",
    "opposite",
    "beside",
    "between",
    "above",
    "below",
    "beneath",
    "underneath",
    "under",
    "over",
    "upper",
    "lower",
    "up",
    "down",
    "upward",
    "downward",
    "forward",
    "backward",
    "north",
    "south",
    "east",
    "west",
    "northeast",
    "northwest",
    "southeast",
    "southwest",
    "northern",
    "southern",
    "eastern",
    "western",
    "clockwise",
    "counterclockwise",
    "anticlockwise",
    "horizontal",
    "vertical",
    "central",
    "inner",
    "outer",
    "midpoint",
}
# nouns too generic for SAM 3, or that only count or group other nouns ("a pair of shoes")
GENERIC_NOUNS = {
    "object",
    "thing",
    "item",
    "stuff",
    "entity",
    "element",
    "instance",
    "example",
    "one",
    "kind",
    "type",
    "sort",
    "category",
    "class",
    "set",
    "pair",
    "group",
    "number",
    "lot",
    "couple",
    "bunch",
    "amount",
    "total",
    "rest",
    "majority",
    "collection",
    "variety",
    "series",
    "piece",
    "part",
    "portion",
    "section",
    "unit",
    "member",
    "option",
    "choice",
    "answer",
    "question",
    "row",
    "following",
    "above",
    "latter",
    "former",
    "fraction",
    "remainder",
}  # not "column": a pillar in indoor scenes
# the recording and the person or robot holding the camera
VIEWPOINT_NOUNS = {
    "video",
    "image",
    "frame",
    "footage",
    "clip",
    "scene",
    "camera",
    "view",
    "viewer",
    "observer",
    "viewpoint",
    "perspective",
    "recording",
    "shot",
    "sequence",
}
# adjectives that do not describe how a thing looks: ordinals, quantifiers, deixis, hedges.
# Comparatives and superlatives are dropped by their part-of-speech tag.
NON_VISUAL_ADJECTIVES = {
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "last",
    "next",
    "previous",
    "final",
    "initial",
    "following",
    "latter",
    "former",
    "other",
    "another",
    "same",
    "different",
    "several",
    "many",
    "few",
    "multiple",
    "various",
    "numerous",
    "single",
    "only",
    "whole",
    "entire",
    "total",
    "overall",
    "both",
    "each",
    "every",
    "all",
    "certain",
    "particular",
    "specific",
    "given",
    "such",
    "own",
    "main",
    "primary",
    "possible",
    "likely",
    "actual",
    "real",
    "approximate",
    "exact",
    "current",
    "visible",
    "invisible",
    "combined",
    "remaining",
    "listed",
    "mentioned",
    "aforementioned",
    "respective",
    "corresponding",
    "relative",
    "absolute",
    "direct",
    "straight",
    "additional",
    "extra",
    "positive",
    "negative",
    "closest",
    "nearest",
    "furthest",
    "farthest",
    "much",
    "more",
    "most",
    "less",
    "least",
    "similar",
    "individual",
    "separate",
    "unique",
    "distinct",
    "average",
    "typical",
    "starting",
}
# plurals that would change meaning if singularised ("glasses" -> "glass")
PLURALIA_TANTUM = {
    "glasses",
    "eyeglasses",
    "sunglasses",
    "scissors",
    "pants",
    "trousers",
    "jeans",
    "shorts",
    "stairs",
    "headphones",
    "earphones",
    "binoculars",
    "pliers",
    "tongs",
    "goggles",
    "clothes",
    "blinds",
    "tweezers",
    "pajamas",
    "pyjamas",
}

NOUNS = {"NOUN", "PROPN"}
# parts of speech that end a noun phrase: a modifier is never separated from its noun by these
# (hyphens excepted), whatever the parse says
PHRASE_BREAKS = {"PUNCT", "CCONJ", "SCONJ", "DET", "ADP", "PRON", "AUX", "PART", "SYM"}
SELF_PRONOUNS = {"i", "you", "we"}
WORD_RE = re.compile(r"[a-z][a-z'-]*[a-z]")  # at least two letters


@dataclass
class Candidate:
    """A noun phrase from the question and whether it becomes a SAM 3 prompt."""

    text: str  # the would-be prompt ("red car")
    keep: bool
    reason: str


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------
@cache
def load_nlp(model=SPACY_MODEL):
    import spacy

    try:
        return spacy.load(model, disable=["ner"])
    except OSError:
        sys.exit(
            f"[prompts] spaCy model '{model}' is not installed: "
            f"python -m spacy download {model}"
        )


@cache
def ensure_wordnet():
    try:
        wn.ensure_loaded()
    except LookupError:
        print("[prompts] downloading WordNet through nltk", file=sys.stderr)
        if not nltk.download("wordnet", quiet=True):
            sys.exit(
                "[prompts] could not download WordNet: python -m nltk.downloader wordnet"
            )
        wn.ensure_loaded()
    return True


# ---------------------------------------------------------------------------
# WordNet tests
# ---------------------------------------------------------------------------
@cache
def _physical_entity():
    return wn.synset("physical_entity.n.01")


@cache
def _unit_of_measurement():
    return wn.synset("unit_of_measurement.n.01")


def _hypernyms(synset):
    return set(synset.closure(lambda s: s.hypernyms() + s.instance_hypernyms()))


@cache
def is_physical(synset):
    lexname = synset.lexname()
    if lexname in PHYSICAL_LEXNAMES:
        return True
    if lexname == "noun.Tops":  # unique beginners: "person", "animal" but also "space"
        return synset == _physical_entity() or _physical_entity() in _hypernyms(synset)
    return False


@cache
def is_physical_group(synset):
    """A group of physical members: "people" -> person, "herd" -> cattle.

    Only used for a word's main sense: most words have some group sense with people as members
    ("order" a religious order, "following" a set of followers).
    """
    return synset.lexname() == "noun.group" and any(
        map(is_physical, synset.member_meronyms())
    )


@cache
def is_unit(synset):
    return _unit_of_measurement() in _hypernyms(synset)


def _lexname(synset):
    return synset.lexname().split(".", 1)[1]


def sense_counts(senses, forms):
    """SemCor frequency of each sense, for whichever of `forms` names it."""
    return [
        max(
            (lem.count() for lem in s.lemmas() if lem.name().lower() in forms),
            default=0,
        )
        for s in senses
    ]


def judge_word(word, lemma):
    """Physical or not, for a single noun. `word` is the text, `lemma` its base form."""
    senses = wn.synsets(word, pos=wn.NOUN) or wn.synsets(lemma, pos=wn.NOUN)
    if not senses:
        return True, "not in WordNet, kept to be safe"
    physical = [
        is_physical(s) or (i == 0 and is_physical_group(s))
        for i, s in enumerate(senses)
    ]
    main_sense = _lexname(senses[0])
    if not any(physical):
        return False, f"no physical sense (main sense: {main_sense})"
    counts = sense_counts(senses, {word, lemma, wn.morphy(word, wn.NOUN)})
    of = "uses"
    if not any(
        counts
    ):  # not in SemCor: WordNet's sense order means little, count each once
        counts, of = [1] * len(senses), "senses"
    total = sum(counts)
    if sum(c for s, c in zip(senses, counts) if is_unit(s)) / total >= MIN_UNIT_SHARE:
        return False, "mostly a unit of measurement"
    if physical[0]:
        return True, f"main sense is physical ({main_sense})"
    share = sum(c for p, c in zip(physical, counts) if p) / total
    if senses[0].lexname() in QUESTION_LEXNAMES and share < MIN_PHYSICAL_SHARE:
        return (
            False,
            f"main sense is {main_sense}, physical senses rare ({share:.0%} of {of})",
        )
    phys = sorted({_lexname(s) for s, p in zip(senses, physical) if p})
    return True, f"has a physical sense ({', '.join(phys)}), kept to be safe"


def judge_compound(forms):
    """Verdict from a multi-word name WordNet knows, or None to fall back on the head noun."""
    for form in forms:
        senses = wn.synsets(form, pos=wn.NOUN)
        if not senses:
            continue
        name = form.replace("_", " ")
        if any(is_physical(s) for s in senses):
            return True, f"'{name}' is a physical thing in WordNet"
        if senses[0].lexname() == "noun.location":
            # multi-word places are concrete ("bus stop", "parking lot"); single-word ones are
            # mostly relations ("side", "area")
            return True, f"'{name}' is a place in WordNet"
        if senses[0].lexname() in QUESTION_LEXNAMES:
            return False, f"'{name}' is {_lexname(senses[0])} in WordNet"
        # neither: e.g. "traffic light" is a signal (communication). Try shorter names, then the head
    return None


# ---------------------------------------------------------------------------
# noun phrases
# ---------------------------------------------------------------------------
def head_lemma(tok, syntactic=False):
    word = tok.text.lower()
    if syntactic:
        return tok.lemma_.lower()
    if word in PLURALIA_TANTUM:
        return word
    lemma = tok.lemma_.lower()
    if not wn.lemmas(
        lemma, pos=wn.NOUN
    ):  # left inflected: a misspelt plural ("bookshelfs"),
        lemma = wn.morphy(lemma, wn.NOUN) or lemma  # a plural tagged as a proper noun
    return lemma


def attached(mod, head):
    """Whether nothing that ends a noun phrase lies between a modifier and its head.

    The parser sometimes links the items of a list ("(table, chair, door, telephone)" -> "door
    telephone") or of a coordination ("the microwave and the trash can").
    """
    lo, hi = sorted((mod.i, head.i))
    return not any(
        t.pos_ in PHRASE_BREAKS and t.text != "-" for t in mod.doc[lo + 1 : hi]
    )


def is_nominal(tok):
    """A noun, or a word the parser uses as one: the head of a noun compound ("the trash can"
    with "can" tagged as a modal verb)."""
    return tok.pos_ in NOUNS or any(
        c.dep_ == "compound" and c.pos_ in NOUNS and attached(c, tok) for c in tok.lefts
    )


def describes_appearance(adj):
    """An adjectival modifier worth passing to SAM 3 ("red", "wooden", "parked")."""
    word = adj.text.lower()
    if adj.tag_ in ("JJR", "JJS", "RBR", "RBS") or any(ch.isdigit() for ch in word):
        return False
    if (
        word in NON_VISUAL_ADJECTIVES
        or word in SPATIAL_WORDS
        or not WORD_RE.fullmatch(word)
    ):
        return False
    # "the most expensive chair"
    return not any(c.lower_ in ("most", "more", "least", "less") for c in adj.children)


def modifiers(head, syntactic=False):
    """Compound nouns and descriptive adjectives in front of `head`, recursively, in text order.

    With `syntactic`, every compound and adjectival modifier that is a word, whatever it means.
    """
    out = []

    def visit(tok):
        for c in tok.lefts:
            word = c.text.lower()
            if not attached(c, tok):
                keep = False
            elif c.dep_ == "compound" and c.pos_ in NOUNS | {"ADJ", "VERB", "NUM", "X"}:
                keep = (syntactic or word not in SPATIAL_WORDS) and re.fullmatch(
                    r"[a-z0-9][a-z0-9'-]*", word
                )
            elif c.dep_ in ("amod", "nmod") or (
                c.dep_ in ("advmod", "npadvmod") and c.pos_ == "ADJ"
            ):
                keep = WORD_RE.fullmatch(word) if syntactic else describes_appearance(c)
            else:
                keep = False
            if keep:
                visit(c)
                out.append(c)

    visit(head)
    return sorted(out, key=lambda t: t.i)


def separators(tokens):
    """What goes between consecutive tokens of a phrase: '-', '' or ' '."""
    doc = tokens[0].doc
    seps = []
    for a, b in pairwise(tokens):
        if b.i == a.i + 2 and doc[a.i + 1].text == "-":
            seps.append("-")
        elif b.i == a.i + 1 and not a.whitespace_:
            seps.append("")
        else:
            seps.append(" ")
    return seps


def render(tokens, lemma):
    words = [t.text.lower() for t in tokens[:-1]] + [lemma]
    seps = separators(tokens) + [""]
    return "".join(w + s for w, s in zip(words, seps)).strip()


def compound_forms(tokens, lemma):
    """WordNet spellings of the multi-word names ending at the head, longest first."""
    forms = []
    for start in range(len(tokens) - 1):
        text = render(tokens[start:], lemma)
        forms.append(text.replace(" ", "_"))
        if len(tokens) - start == 2 and " " in text:  # "lamp post" -> "lamppost"
            forms.append(text.replace(" ", ""))
    return forms


def is_self_reference(tok):
    """'You are a robot ...', 'I am the driver': the questioner, not something in the scene."""
    verb = tok.head
    return (
        tok.dep_ == "attr"
        and verb.lemma_ == "be"
        and any(
            c.dep_ in ("nsubj", "nsubjpass") and c.lower_ in SELF_PRONOUNS
            for c in verb.children
        )
    )


def judge_phrase(tokens):
    """(keep, reason, lexicalized): lexicalized when WordNet judged the multi-word name as a whole."""
    head = tokens[-1]
    word, lemma = head.text.lower(), head_lemma(head)
    if not (WORD_RE.fullmatch(word) or WORD_RE.fullmatch(lemma)):
        return False, "not a word", False
    if word in SPATIAL_WORDS or lemma in SPATIAL_WORDS:
        return False, "spatial relation word", False
    bare = not any(t.dep_ == "compound" or t.pos_ in NOUNS for t in tokens[:-1])
    if bare and lemma in GENERIC_NOUNS:
        return False, "too generic", False
    if bare and lemma in VIEWPOINT_NOUNS:
        return False, "refers to the video or the viewer", False
    verdict = judge_compound(compound_forms(tokens, lemma)) if len(tokens) > 1 else None
    if verdict:
        return *verdict, True
    return *judge_word(word, lemma), False


def join_wordnet_compounds(doc):
    """Make a word a compound of the next noun when WordNet names the two together but the
    parser split them.

    The parser sometimes reads the first word as the end of one phrase and the noun as the start
    of the next ("closer to the trash | can ..."), the first word as a verb taking the noun as
    its object ("cutting board", "living room"), or the noun as a verb nothing depends on ("the
    closest to the trash can?", with "can" a modal).
    """
    for a, b in pairwise(doc):
        if not (a.whitespace_ and a.sent == b.sent and a.head != b):
            continue
        dangling = (
            a.pos_ in NOUNS and b.pos_ in ("AUX", "VERB") and not list(b.children)
        )
        if not (a.pos_ in NOUNS | {"VERB", "ADJ"} and (b.pos_ in NOUNS or dangling)):
            continue
        words = (a.text.lower(), head_lemma(b))
        if not any(
            wn.synsets(f, pos=wn.NOUN) for f in ("_".join(words), "".join(words))
        ):
            continue
        if b.head == a or dangling:  # the noun takes the first word's place in the tree
            b.head, b.dep_ = (b if a.head == a else a.head), a.dep_
        elif a.is_ancestor(b):
            continue  # re-attaching would make a loop
        a.head, a.dep_ = b, "compound"


def analyse(question, model=SPACY_MODEL, syntactic=False):
    """Every noun phrase of the question, with whether it becomes a SAM 3 prompt and why.

    With `syntactic`, only the parse: every noun phrase is kept (see the module docstring).
    """
    if not syntactic:
        ensure_wordnet()
    text = question.replace("\\n", " ")  # a literal "\n" copied from a dataset dump
    text = re.sub(
        r"(?<=\w)\(e?s\)", "", text
    )  # "chair(s)", "shoes(s)" -> "chair", "shoes"
    # answer options and numbered steps ("A. bench B. tree", "(1) ..."): end the phrase before
    # each marker, or the parser reads the options as one long compound
    text = re.sub(r"(?:^|(?<=\s))\(?(?:[A-H]|\d{1,2})[.)](?=\s)", ";", text)
    doc = load_nlp(model)(text)
    if not syntactic:
        join_wordnet_compounds(doc)

    heads = [t for t in doc if is_nominal(t)]
    mods = {t.i: modifiers(t, syntactic) for t in heads}
    in_phrase = {m.i for ms in mods.values() for m in ms}  # modifiers of another noun

    candidates = []
    for tok in heads:
        if tok.i in in_phrase:
            continue
        if syntactic:
            lemma = head_lemma(tok, syntactic=True)
            word_like = WORD_RE.fullmatch(tok.text.lower()) or WORD_RE.fullmatch(lemma)
            candidates.append(
                Candidate(
                    render(mods[tok.i] + [tok], lemma),
                    bool(word_like),
                    "noun phrase (syntactic mode)" if word_like else "not a word",
                )
            )
            continue
        if is_self_reference(tok):
            candidates.append(
                Candidate(tok.text.lower(), False, "the questioner, not the scene")
            )
            continue
        tokens = mods[tok.i] + [tok]
        while True:
            keep, reason, lexicalized = judge_phrase(tokens)
            candidates.append(
                Candidate(render(tokens, head_lemma(tokens[-1])), keep, reason)
            )
            if keep or lexicalized:  # "eye level" is a height, not an eye
                break
            # head dropped: try what modifies it ("the table height" -> "table")
            nouns = [i for i, t in enumerate(tokens[:-1]) if t.pos_ in NOUNS]
            if not nouns:
                break
            tokens = tokens[: nouns[-1] + 1]
    return candidates


def kept_prompts(candidates):
    return list(
        dict.fromkeys(c.text for c in candidates if c.keep)
    )  # in order, no duplicates


def main(question, model=SPACY_MODEL, syntactic=False):
    """SAM 3 prompts for a question: the physical things it names, in order of appearance.

    With `syntactic`, every noun phrase it contains instead (see the module docstring).
    """
    return kept_prompts(analyse(question, model, syntactic))


def cli():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("questions", nargs="*", help="questions to extract prompts from")
    p.add_argument(
        "--file",
        help="text file with one question per line (a line break inside a "
        "question can be written as \\n)",
    )
    p.add_argument(
        "--explain",
        action="store_true",
        help="list every candidate noun phrase with the reason it was kept or dropped",
    )
    p.add_argument(
        "--syntactic",
        action="store_true",
        help="skip everything based on meaning (WordNet, word lists): every noun phrase, "
        "reduced to its head noun and its compound and adjectival modifiers, is a prompt",
    )
    p.add_argument(
        "--model", default=SPACY_MODEL, help=f"spaCy pipeline (default {SPACY_MODEL})"
    )
    args = p.parse_args()

    questions = list(args.questions)
    if args.file:
        with open(args.file, encoding="utf-8") as fh:
            questions += [line.strip() for line in fh if line.strip()]
    if not questions:
        p.error("give at least one question, or --file")

    for q in questions:
        candidates = analyse(q, args.model, args.syntactic)
        if args.explain:
            print(f"\n{q}")
            for c in candidates:
                print(f"  {'keep' if c.keep else 'drop'}  {c.text:28s} {c.reason}")
            print(f"  -> {', '.join(kept_prompts(candidates))}")
        else:
            print(",".join(kept_prompts(candidates)))


if __name__ == "__main__":
    cli()
