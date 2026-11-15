# OpenVocab-4D comparison: flat tables

Built from the per-scene folders next to this one by `tasks/openvocab_adapter/export_tables.py` (standard library
only; rerun it after new results). Run of 2026-11-15, one B200: 10 scenes listed in `hpc/scenes.tsv`, frame selection
`half` (every other frame of the 128, i.e. 64), plus a 5-frame test (`smoke5`) on two of them. SAM 3 masks come from
`facebook/sam3` at revision 3c879f3 through transformers (`sam3_infer.py --backend transformers`).

Every table has `dataset` and `scene`. `selection` is the frame selection; `run` is `adapted` (the saved VGGT outputs
from `data/vggt/`, converted by `adapt_vggt.py`) or `native` (OpenVocab-4D running its own VGGT on the same frames).
Both runs use the same SAM 3 masks. Positions are in each run's own coordinates: the camera of the first frame of
the reconstruction, x right, y down, z forward, in VGGT's units (not metres). The two runs' coordinates differ by a
rotation, translation and scale, so compare positions across runs only after alignment (`comparisons.csv` has the
aligned errors).

ARKitScenes 41069025 has no `half` result: the adapter stopped because frame 0's camera deviates from the origin by
1.4e-3, above its 1e-3 limit. Its SAM 3 detections are in `detections.csv`.

## objects.csv: the scene sheets, one row per object

| Column | Meaning |
|---|---|
| `object_id` | row number within that scene sheet |
| `concept` | the SAM 3 prompt (object name) the object was found with |
| `track_id` | the tracker's id for the object, unique per concept within a run |
| `x`, `y`, `z` | position: per-axis median of the object's 3D points |
| `size_x`, `size_y`, `size_z`, `size` | extent of the object's points along each axis, and their product. Axis-aligned, not aligned with the object |
| `num_frames` | number of frames the object is visible in |
| `frames_seen` | those frames, `;`-separated, as positions in the selection (0..63 for `half`); `cameras.csv` maps them to frame numbers |

## cameras.csv: one row per frame and run

| Column | Meaning |
|---|---|
| `frame` | position in the selection (matches `frames_seen`) |
| `export_frame` | frame number in the 128-frame VGGT export (0..127) |
| `video_frame` | frame number in the original video |
| `x`, `y`, `z` | camera centre |
| `qw`, `qx`, `qy`, `qz` | camera orientation: the rotation from camera axes to world axes, as a unit quaternion |
| `forward_x`, `forward_y`, `forward_z` | viewing direction (camera +z) in world coordinates |

## comparisons.csv: the adapted vs native comparison, one row per measured quantity

| Column | Meaning |
|---|---|
| `verdict` | MATCHES if every criterion passes, DIFFERS if not |
| `kind` | `criterion` (a pass mark that decides the verdict), `measure` (a measured value) or `check` (a sanity check) |
| `name` | what is measured |
| `status` | PASS, FAIL, MEASURED or n/a |
| `value`, `limit` | the value, and the pass mark for criteria. Lists and dicts (e.g. per-object errors) are JSON |
| `detail` | free-text note from `compare.py` |

## detections.csv: SAM 3, one row per frame and prompt

| Column | Meaning |
|---|---|
| `frame` | frame number in the 128-frame VGGT export (`export_frame` in `cameras.csv`) |
| `prompt` | the object name given to SAM 3 |
| `detections` | masks with score at least `min_score` |
| `min_score`, `model` | the score threshold when masks were saved (tracking then uses 0.55), and the model |

The masks, depth maps and point clouds stay out of git, in `tasks/openvocab_adapter/out/`: they are derived from the
scenes' video frames, which ScanNet's terms do not allow sharing.
