# OpenVocab-4D: the saved VGGT outputs (adapted) vs its own VGGT (native)

10 scenes, 12 comparisons: 11 DIFFERS, 1 no result. Pass marks are provisional (`CRITERIA` in compare.py).

Columns: objects found (adapted / native); share of tracked pixels with the same identity (pass >= 90%); median 3D object position error as a share of the scene size (pass <= 5%); share of scene-sheet rows in both sheets (pass >= 80%); median scene-sheet position error (pass <= 5%).

| Scene | Frames | Verdict | Objects | Same track | Position error | Sheet rows | Sheet error | FAIL checks |
|---|---|---|---|---|---|---|---|---|
| arkitscenes/41069025 | - | no result | | | | | | `[adapter] 1 check(s) failed, nothing written: ["cameras: world frame is frame 0'` |
| arkitscenes/41159525 | smoke5 | **DIFFERS** | 1 / 2 | 46.6% | 1.8% | 66.7% | 1.1% |  |
| arkitscenes/41159525 | half | **DIFFERS** | 10 / 5 | 48.4% | 0.8% | 66.7% | 0.7% |  |
| arkitscenes/42444976 | half | **DIFFERS** | 6 / 5 | 40.2% | 1.0% | 54.5% | 0.9% |  |
| scannet/scene0025_01 | smoke5 | **DIFFERS** | 4 / 4 | 72.8% | 0.6% | 100.0% | 0.9% |  |
| scannet/scene0025_01 | half | **DIFFERS** | 21 / 21 | 64.8% | 0.7% | 95.2% | 0.7% |  |
| scannet/scene0046_01 | half | **DIFFERS** | 8 / 11 | 54.4% | 1.1% | 73.7% | 1.0% |  |
| scannet/scene0149_00 | half | **DIFFERS** | 11 / 15 | 28.9% | 1.0% | 76.9% | 0.9% |  |
| scannet/scene0207_02 | half | **DIFFERS** | 9 / 7 | 59.6% | 0.6% | 87.5% | 0.6% |  |
| scannetpp/09c1414f1b | half | **DIFFERS** | 21 / 17 | 45.3% | 0.9% | 68.4% | 0.9% |  |
| scannetpp/0d2ee665be | half | **DIFFERS** | 16 / 19 | 42.1% | 1.3% | 80.0% | 1.3% |  |
| scannetpp/1ada7a0617 | half | **DIFFERS** | 16 / 10 | 39.3% | 1.2% | 61.5% | 1.3% |  |
