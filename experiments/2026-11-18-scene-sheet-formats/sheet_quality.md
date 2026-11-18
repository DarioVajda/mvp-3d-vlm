# Pipeline vs gold scene sheets

<!-- Written by sheet_quality.py in this folder; edit that, not this file. -->

Source: `scene_sheets_75` (Dario's run of `run_sheets.sh`: 75 scenes, 2,728 questions), read
only. Regenerate with `PYTHONPATH=src python experiments/2026-11-18-scene-sheet-formats/sheet_quality.py`.

**Named objects** are a question's prompts from `question.json`, without the prompt `room`. The extractor adds `room`
to every counting question ("in this room") and uses it as the only prompt for room-size questions. Room-size
questions (75) name no object, so sections 1, 3 and 4 leave them out. That leaves 2,653
questions and 7,587 (question, named object) pairs. An object is **found** if its prompt has at least one
row in the sheet. The gold sheet's rows are VSI-Bench's labelled boxes for the class the prompt was matched to
(`sheet.json`, `matched_classes`).

## Summary

- **Pipeline sheets miss about a quarter of the objects the questions ask about.** 73.4% of named
  objects have at least one pipeline row, against 97.4% in gold. Only 48.8% of questions
  have every named object in the pipeline sheet, against 96.1% in gold.
- **One object usually becomes several rows.** 69% of pipeline rows are seen in only 1 of the 16
  frames, and 87% in at most 2. Re-identification rarely joins views of the same object, so large
  objects are over-counted (door, bed, table, pillow, sofa). The worst, door, averages +3.8
  rows per scene, with up to 18 rows for one object.
- **Counts agree only 28% of the time.** For the 5,906 pairs where gold has exactly
  one object, which covers most VSI-Bench questions, the pipeline has exactly one row in
  31% of cases. It has none in 29% and two or more
  in 40%, which leaves the model to guess which row the question
  means.
- **Empty sheets come from small or rare objects and a few bad scenes.** 205 of the 208
  empty pipeline sheets have a non-empty gold sheet. 122 of the 208 are object-size questions
  (30% of them), which usually name a single, often small, object.
- **Relative sizes are roughly right but noisy.** Across a question's objects, the pipeline's ratio of largest to
  smallest object is within 1.5× of gold's in 59% of questions. It picks the same largest object
  in 68% (chance: 38%). Each scene's pipeline-to-metre scale varies by a median
  factor of 1.5 between objects of the same scene.
- **Gold has its own gaps, all in prompt matching.** 198 of the 198 gold
  misses are prompts matched to no class. 140 are route-planning landmarks
  ("wall", "window", "sink", "door"). The rest come from two extraction bugs: "ceiling light" split into `ceiling` + `light`
  when it ends the question, and "shoes" singularised to `shoe` while the class is `shoes`.

## 1. Are the named objects in the sheets?

Share of (question, named object) pairs with at least one row, and share of questions with every named object:

| Question type | Pairs | Pipeline | Gold | Questions, all found (pipeline) | Questions, all found (gold) |
|---|---:|---:|---:|---:|---:|
| Appearance order | 1,942 | 77.0% | 99.3% | 41.9% | 97.1% |
| Absolute distance | 856 | 70.8% | 99.5% | 51.6% | 99.1% |
| Counting | 223 | 84.8% | 98.7% | 84.8% | 98.7% |
| Relative direction, easy | 469 | 72.9% | 98.5% | 41.9% | 96.8% |
| Relative direction, medium | 759 | 71.0% | 100.0% | 38.7% | 100.0% |
| Relative direction, hard | 798 | 72.9% | 100.0% | 44.0% | 100.0% |
| Relative distance | 1,902 | 70.7% | 98.5% | 23.7% | 94.7% |
| Object size | 404 | 69.8% | 99.8% | 69.8% | 99.8% |
| Route planning | 234 | 81.2% | 40.2% | 50.0% | 3.4% |
| **All** | **7,587** | **73.4%** | **97.4%** | **48.8%** | **96.1%** |

- Of the 7,587 pairs, 1,956 are in gold but not the pipeline, 138 in
  the pipeline but not gold, and 60 in neither.
- **Questions that name more objects lose more.** Relative distance names 5 objects, and only
  24% of those questions have all of them in the pipeline sheet.
- **By dataset**, pipeline coverage is 82.9% on ScanNet, 73.6% on ARKitScenes, 60.1% on ScanNet++.
- **By scene**, the median is 75%, and a quarter of scenes are below
  62%. The worst scenes are `arkitscenes/47429922` (9%), `scannetpp/fb5a96b1a2` (26%), `arkitscenes/44358499` (27%), `scannetpp/3db0a1c8f3` (38%), `scannetpp/3f15a9266d` (40%).
- **Classes the pipeline misses most**, counting each (scene, class) once among those gold has, with at least five
  scenes: heater 0 of 13, ceiling light 1 of 14, mirror 2 of 7, computer mouse 4 of 12, plant 3 of 9, stove 5 of 12, microwave 3 of 6, computer tower 3 of 5, telephone 9 of 14.
- **Classes it almost always finds** (90% or more, at least eight scenes): bed, pillow, whiteboard, door, table, chair, backpack.
- **Gold's misses** are the 198 unmatched prompts. 140 are route-planning
  landmarks, which are free text rather than dataset classes. That makes route planning the only type where gold
  covers less than the pipeline. The others are `shoe` (38), `ceiling` (10), `light` (10).

## 2. The 208 empty pipeline sheets

208 questions (57 scenes) have no pipeline rows at all.

| Question type | Empty | Share of type | Gold sheet has objects |
|---|---:|---:|---:|
| Object size | 122 | 30.2% | 121 |
| Absolute distance | 43 | 10.0% | 43 |
| Relative direction, medium | 15 | 5.9% | 15 |
| Relative direction, hard | 13 | 4.9% | 13 |
| Relative direction, easy | 8 | 5.2% | 8 |
| Relative distance | 4 | 1.1% | 4 |
| Room size | 2 | 2.7% | 0 |
| Route planning | 1 | 1.7% | 1 |
| **All** | **208** | **7.6%** | **205** |

- **Gold has objects for 205 of the 208.** The exceptions: 3804 (room size, whose gold sheet is always empty); 3793 (room size, whose gold sheet is always empty); 2375 (object size, `shoe` matched to no gold class).
- **Prompts per empty sheet:** 124 with 1, 44 with 2, 36 with 3, 4 with 5 prompts. So these are mostly single-object questions where that one
  object was never segmented.
- **Most common single prompts:** heater 9, stove 7, tv 7, ceiling light 6, sofa 5, mirror 5, computer mouse 5, telephone 5, refrigerator 4, stool 4, trash bin 4, microwave 3, radiator 3, window 3, kettle 3, keyboard 3, bookshelf 3. The empty multi-prompt sheets (for example
  `sofa+chair+tv`, `sofa+tv+chair`) mostly come from the bad scenes below, where even common objects were missed.
- **Bad scenes:** 51 of the 208 come from three scenes: `scannetpp/fb5a96b1a2` (20, 53% of its questions), `scannetpp/3db0a1c8f3` (17, 23% of its questions), `arkitscenes/47429922` (14, 74% of its questions).
- **No empty sheets in appearance order or counting.** Counting always has the `room` prompt, and appearance
  order has four prompts, so at least one usually has rows.
- **What the eval sees:** every format renders an empty sheet as "The detector did not find the objects named in the question. This does not mean they are absent from the room."

## 3. Counts of each named class

For the 7,389 (question, named class) pairs that have gold rows, pipeline rows minus gold rows:

| Question type | Pairs | Same count | Pipeline more | Pipeline fewer, not 0 | Pipeline 0 | Mean difference |
|---|---:|---:|---:|---:|---:|---:|
| Appearance order | 1,928 | 28.3% | 42.8% | 6.0% | 22.8% | +0.8 |
| Absolute distance | 852 | 30.0% | 40.5% | 0.6% | 28.9% | +0.7 |
| Counting | 220 | 17.3% | 44.1% | 23.6% | 15.0% | +0.4 |
| Relative direction, easy | 462 | 33.5% | 39.8% | 0.0% | 26.6% | +0.6 |
| Relative direction, medium | 759 | 29.5% | 41.1% | 0.4% | 29.0% | +0.8 |
| Relative direction, hard | 798 | 29.4% | 43.2% | 0.3% | 27.1% | +0.9 |
| Relative distance | 1,873 | 25.0% | 38.5% | 7.5% | 29.0% | +0.5 |
| Object size | 403 | 29.0% | 40.4% | 0.5% | 30.0% | +0.7 |
| Route planning | 94 | 28.7% | 53.2% | 3.2% | 14.9% | +1.0 |
| **All** | **7,389** | **28.0%** | **41.2%** | **4.4%** | **26.5%** | **+0.7** |

- **The usual over-count is about double.** When the pipeline has more rows, it has a median of
  2 extra (mean 2.6, 90th percentile 6). When the
  pipeline finds the class at all, the median ratio of pipeline to gold rows is 2.0.
- **The usual under-count is the object missing entirely.** Only 4.4% of pairs have fewer rows
  than gold without being zero.
- **Unique objects:** gold has exactly one object in 5,906 pairs (80%), as
  VSI-Bench intends for every type except counting. There the pipeline has 0 rows in 1,728
  (29%), 1 in 1,840 (31%), 2 in
  906, 3 in 788 and 4 or more in 644.
- **Counting questions:** 38 of 220 are exact. In counting sheets
  2,155 of the 2,892 rows (75%) are `room` rows, pieces of the
  room found in single frames. That noise sits next to the rows that matter.
- **By class**, mean pipeline minus gold per (scene, class), classes in at least 8 scenes:
  - over: door +3.8 (max 18 rows), bed +3.3, table +1.6, pillow +1.6, sofa +1.2, whiteboard +1.1;
  - under: ceiling light −2.9, plant −1.7, heater −1.3, computer mouse −1.2, monitor −0.8.
- **Why:** of the 1,907 pipeline rows for named classes (each scene and class once),
  69% are seen in only 1 of the 16 frames and 87% in at most 2. The median
  is 1. Each view of a large object becomes its own track.

## 4. Relative sizes within a question

Units differ (VGGT units vs metres) and so do axes, so this compares sizes within each sheet. Each class's size is
the longest side of its box (the `size_x/y/z` maximum), taken as the median over its rows. Using the largest row
instead gives the numbers after the slash. The sample is the 1,597 questions with at least two named classes
found in both sheets.

- **Largest-to-smallest ratio:** the median is 2.28× in the pipeline vs 2.83× in
  gold (2.52× vs 2.88×). The pipeline compresses size differences, because big
  objects are split into fragments.
- **How close the ratio is:** the median |log₂(pipeline ratio / gold ratio)| is 0.51, a factor of
  1.43 (0.42, 1.34×). It is within 1.5× of gold in
  59% of questions (61%) and within 2× in 83%
  (87%). The pipeline's ratio is the smaller one in 60% of questions.
- **Correlation:** the log ratios correlate at r = 0.66 (0.70).
- **Same largest class** in 68% of questions (76%), and **same smallest** in
  72% (74%). Picking at random would give 38%.
- **Rank correlation** over 944 questions with three or more classes: Spearman median 0.70
  (0.80).
- **By type, same largest class:** absolute distance 77%, relative direction, hard 75%, relative direction, medium 67%, relative direction, easy 65%, appearance order 64%, relative distance 63%.
- **Scale within a scene:** in 71 scenes with three or more classes in both sheets, each class's
  pipeline-to-gold size ratio varies with a median standard deviation of 0.57 in log₂, a factor of about
  1.5. The largest and smallest ratios in a scene differ by a median of 2.8×. So one
  scale factor per scene, such as from a known object, would still leave sizes off by about 1.5×.

## 5. Questions where pipeline and gold disagree most

For each of three question types, this takes the question with the largest total |pipeline − gold| over its named
classes, each from a different scene. The sheets are shown as the eval renders them in `csv`, without the preamble.
Pipeline rows are in VGGT units with the camera axes; gold rows are in metres, z up.

### 3559, object size, `scannet/scene0231_00`

*What is the length of the longest dimension (length, width, or height) of the door, measured in centimeters?* Answer: 165.

Pipeline / gold rows per named class: door 18 / 1. 12 of the 18 pipeline rows of named classes are seen in a single frame.

<table>
<tr><th>Pipeline</th><th>Gold</th></tr>
<tr>
<td valign="top"><pre>
id,label,x,y,z,size_x,size_y,size_z,first_seen,num_frames
door_1,door,-0.546,-0.113,1.127,0.431,0.554,0.439,5.8,2
door_2,door,-0.571,-0.425,1.257,0.045,0.151,0.403,5.8,1
door_7,door,-0.054,0.075,0.869,0.540,0.251,0.403,5.8,2
door_9,door,-0.565,-0.061,1.545,0.075,0.289,0.282,5.8,2
door_11,door,0.095,0.255,0.264,0.148,0.226,0.094,29.8,1
door_12,door,0.122,0.278,0.266,0.128,0.229,0.072,29.8,1
door_13,door,0.669,0.273,0.420,0.108,0.233,0.384,79.3,1
door_14,door,0.786,-0.107,0.381,0.259,0.277,0.390,79.3,2
door_17,door,0.735,0.086,1.121,0.399,0.315,0.485,79.3,2
door_19,door,-0.017,0.037,1.235,0.282,0.188,0.075,104.1,1
door_20,door,-0.282,-0.054,1.600,0.404,0.189,0.111,104.1,1
door_22,door,-0.576,0.015,1.368,0.025,0.218,0.144,104.1,1
door_25,door,0.847,-0.337,1.188,0.034,0.245,0.182,128.8,2
door_26,door,0.755,-0.099,0.230,0.033,0.208,0.116,128.8,1
door_27,door,0.761,0.009,1.425,0.024,0.247,0.102,128.8,1
door_29,door,0.665,0.245,0.585,0.468,0.181,0.063,153.6,1
door_30,door,0.702,0.167,0.831,0.393,0.147,0.092,153.6,1
door_31,door,0.374,-0.557,1.496,0.265,0.227,0.089,153.6,1
</pre></td>
<td valign="top"><pre>
id,label,x,y,z,size_x,size_y,size_z
door_1,door,0.002,2.103,-0.200,0.417,0.890,1.695
</pre></td>
</tr>
</table>

### 2851, appearance order, `scannetpp/d755b3d9d8`

*What will be the first-time appearance order of the following categories in the video: keyboard, plant, door, monitor?* Answer: D.

Pipeline / gold rows per named class: keyboard 1 / 2, plant 0 / 3, door 12 / 1, monitor 0 / 3. No pipeline rows for plant, monitor, which gold has. 11 of the 13 pipeline rows of named classes are seen in a single frame.

<table>
<tr><th>Pipeline</th><th>Gold</th></tr>
<tr>
<td valign="top"><pre>
id,label,x,y,z,size_x,size_y,size_z,first_seen,num_frames
door_1,door,0.153,-0.052,0.564,0.282,0.441,0.443,4.7,2
door_2,door,-0.478,-0.094,0.765,0.056,0.184,0.126,4.7,1
door_3,door,-0.405,-0.223,0.774,0.135,0.277,0.282,4.7,1
door_4,door,-0.410,-0.520,0.500,0.134,0.492,0.435,4.7,1
door_5,door,-0.537,-0.415,0.415,0.141,0.487,0.456,4.7,2
door_6,door,-0.879,-0.184,0.102,0.021,0.388,0.345,4.7,1
door_7,door,-0.669,-0.142,0.460,0.148,0.137,0.139,4.7,1
door_8,door,-0.872,-0.108,0.183,0.021,0.280,0.235,23.8,1
door_9,door,-0.931,-0.243,-0.026,0.139,0.260,0.248,43.6,1
door_10,door,-0.967,0.015,0.170,0.118,0.536,0.519,63.4,1
door_11,door,0.068,-0.281,0.457,0.277,0.620,0.553,83.2,1
door_12,door,-0.445,-0.546,0.417,0.067,0.436,0.367,83.2,1
keyboard_1,keyboard,-0.385,0.738,-0.486,0.127,0.089,0.102,83.2,1
</pre></td>
<td valign="top"><pre>
id,label,x,y,z,size_x,size_y,size_z
door_1,door,0.137,1.789,1.067,0.098,0.943,1.989
keyboard_1,keyboard,2.352,3.321,0.819,0.461,0.209,0.054
keyboard_2,keyboard,3.447,3.423,0.818,0.437,0.213,0.073
monitor_1,monitor,3.953,0.545,1.060,0.650,0.329,0.542
monitor_2,monitor,2.356,3.468,1.040,0.619,0.161,0.472
monitor_3,monitor,3.478,3.423,1.079,0.658,0.343,0.504
plant_1,plant,4.630,2.194,1.613,0.977,1.391,1.435
plant_2,plant,4.798,2.586,1.183,0.385,0.290,0.430
plant_3,plant,2.663,3.542,0.947,0.272,0.293,0.300
</pre></td>
</tr>
</table>

### 2514, counting, `scannetpp/fb5a96b1a2`

*How many ceiling light(s) are in this room?* Answer: 12.

Pipeline / gold rows per named class: ceiling light 0 / 12. No pipeline rows for ceiling light, which gold has. The pipeline sheet also has 10 `room` rows, from the extra prompt.

<table>
<tr><th>Pipeline</th><th>Gold</th></tr>
<tr>
<td valign="top"><pre>
id,label,x,y,z,size_x,size_y,size_z,first_seen,num_frames
room_1,room,-0.086,0.095,0.584,1.019,0.733,0.859,4.4,1
room_2,room,-0.462,0.300,0.300,1.525,1.151,1.431,4.4,2
room_3,room,0.257,-0.262,0.349,0.598,0.971,0.914,22.8,1
room_4,room,-0.831,0.178,0.253,1.713,1.347,1.222,41.6,2
room_5,room,-1.125,-0.833,0.403,1.451,0.450,0.795,60.5,1
room_6,room,-1.189,-0.976,0.727,0.531,0.430,0.373,79.4,1
room_7,room,0.164,-0.270,-0.053,0.754,1.255,1.152,79.4,1
room_8,room,0.004,-0.212,0.648,0.895,0.931,1.268,98.3,1
room_9,room,-0.309,-0.197,0.689,1.102,0.944,1.238,117.2,1
room_10,room,-1.432,0.554,-0.080,0.546,0.602,0.940,136.1,1
</pre></td>
<td valign="top"><pre>
id,label,x,y,z,size_x,size_y,size_z
ceiling_light_1,ceiling light,5.086,1.553,2.605,0.334,0.277,0.269
ceiling_light_2,ceiling light,3.150,1.046,2.591,0.260,0.267,0.249
ceiling_light_3,ceiling light,4.748,2.470,2.601,0.317,0.292,0.259
ceiling_light_4,ceiling light,2.366,1.295,2.664,0.306,0.294,0.262
ceiling_light_5,ceiling light,1.838,3.334,2.602,0.328,0.317,0.263
ceiling_light_6,ceiling light,0.967,4.976,2.594,0.276,0.262,0.244
ceiling_light_7,ceiling light,3.606,5.796,2.600,0.351,0.276,0.270
ceiling_light_8,ceiling light,4.226,4.001,2.597,0.372,0.328,0.341
ceiling_light_9,ceiling light,4.242,6.233,2.625,0.295,0.307,0.309
ceiling_light_10,ceiling light,4.848,5.358,2.592,0.266,0.263,0.238
ceiling_light_11,ceiling light,6.311,3.187,2.613,0.298,0.207,0.323
ceiling_light_12,ceiling light,2.527,2.184,2.598,0.206,0.241,0.253
</pre></td>
</tr>
</table>

## What this means for the format experiment

- **The pipeline sheets are a noisy signal.** A format that helps the model use them may still hurt overall, because
  in 51% of questions the sheet is missing a named object, and for 40% of unique objects it gives several candidates. The format comparison (B–F against
  each other) is still fair, since all five carry the same content. The comparison with A (video only) mixes format
  with sheet quality.
- **Gold sheets separate the two.** Conditions G–K give the same formats with gold content, which shows how much of
  any gap is the sheet's content rather than its format.
- **Types to read with care:**
  - object size: 30% of sheets are empty, and sizes are in VGGT units;
  - appearance order: 58% of questions are missing at least one object;
  - counting: the right count in only 17%, and 75% of rows
    are `room` noise.
