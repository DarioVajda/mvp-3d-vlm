"""VG-LLM's VSI-Bench task functions, unchanged, plus a process_docs that keeps one contiguous slice of the questions
(np.array_split over the order process_docs returns), chosen by VSI_SHARD=<index>/<number of shards>. merge_shards.py
joins the logged per-question scores and aggregates them with VG-LLM's vsibench_aggregate_results. Within a shard that
function can fail, since a contiguous slice may lack a question type it expects (object_rel_direction_easy/medium), and
lmms_eval aggregates before saving the per-question outputs; vsibench_aggregate_shard therefore returns NaN instead.
The per-shard number is not used."""
import os

import numpy as np
from lmms_eval.tasks.vsibench.utils import (process_docs, vsibench_aggregate_results, vsibench_doc_to_text,  # noqa: F401
                                            vsibench_doc_to_visual, vsibench_process_results)


def process_docs_shard(dataset):
    dataset = process_docs(dataset)
    i, n = map(int, os.environ["VSI_SHARD"].split("/"))
    part = np.array_split(np.arange(len(dataset)), n)[i]
    print(f"VSI_SHARD {i}/{n}: rows {part[0]}-{part[-1]} of {len(dataset)}", flush=True)
    return dataset.select(part.tolist())


def vsibench_aggregate_shard(results):
    try:
        return vsibench_aggregate_results(results)
    except KeyError as e:
        print(f"shard lacks question type for {e}; per-shard score skipped", flush=True)
        return float("nan")
