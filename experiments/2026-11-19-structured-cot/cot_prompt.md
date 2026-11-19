# The structured chain-of-thought prompt

<!-- Generated from src/mvp/benchmarks/vsibench.py (cot_prompt, COT_STEPS, COT_HINTS, COT_ANSWER) at commit 6642c48 on scene_sheet_formats, added in 50fe543. The code is the source; regenerate this file rather than editing it. -->

Conditions B and D use the eval's `--reasoning cot` (src/mvp/benchmarks/vsibench.py, `cot_prompt`), added in commit `50fe543`. The instructions replace lmms-eval's last line ("Answer with the option's letter from the given choices directly." / "Please answer the question using a single word or phrase."). They are the general steps, then one hint for the question's task, then the answer line that `score_final` reads. With a sheet, the first step adds "and in the scene information above". Gold sheets add nothing else; the pipeline-only notes (`COT_PIPELINE`, `COT_SCALE`) are not used in this experiment.

## Steps, without a sheet (B)

> Think it through step by step before you answer. First, say which objects or places the question is about and find them in the video. Then write down the facts you need, such as positions, distances, sizes, counts or the order in which things appear, and work out the answer from them one step at a time, showing any arithmetic. Before answering, check that the result agrees with what the video shows.

## Steps, with the gold sheet (D)

> Think it through step by step before you answer. First, say which objects or places the question is about and find them in the video and in the scene information above. Then write down the facts you need, such as positions, distances, sizes, counts or the order in which things appear, and work out the answer from them one step at a time, showing any arithmetic. Before answering, check that the result agrees with what the video shows.

## Hint per task

- `object_counting`: Count each physical object once, even if it appears in many frames, and remember that some may be out of view in a given frame.
- `object_abs_distance`: Find where each of the two objects is, then work out the straight-line distance between them in metres, measured as the question says.
- `object_size_estimation`: Work out the object's length, width and height, and give the longest of the three in centimetres.
- `room_size_estimation`: Work out the room's length and width from its walls, the furniture along them and how far the camera moves, and multiply them to get the floor area in square metres; for several rooms, add up their areas.
- `object_rel_distance`: Estimate the distance from the reference object to each of the options, then pick the one the question asks for.
- `object_rel_direction`: Picture yourself standing where the question puts you and facing the way it says. Work on the floor plan, ignoring height: decide whether the target is in front of you or behind you and whether it is on your left or your right, then pick the matching option.
- `route_planning`: Walk the route one step at a time from the given start and facing: at each step say which way you face, and choose the turn that points you towards the next object.
- `obj_appearance_order`: Find the first frame in which each object appears, and order the objects by it.

## Answer line

- multiple choice: End your reply with a last line of the form "Answer: X", where X is only the letter of the correct option.
- numeric: End your reply with a last line of the form "Answer: X", where X is only a number, in the unit the question asks for.

## Example: absolute distance with the gold sheet (D), as the prompt ends

```
Think it through step by step before you answer. First, say which objects or places the question is about and find them in the video and in the scene information above. Then write down the facts you need, such as positions, distances, sizes, counts or the order in which things appear, and work out the answer from them one step at a time, showing any arithmetic. Before answering, check that the result agrees with what the video shows. Find where each of the two objects is, then work out the straight-line distance between them in metres, measured as the question says.
End your reply with a last line of the form "Answer: X", where X is only a number, in the unit the question asks for.
```
