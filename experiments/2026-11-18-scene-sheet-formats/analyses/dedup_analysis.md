## 1. Overall (dev200; dev results)

VSI-Bench overall: A_video 41.99, pipeline_v3 36.14, pipeline_v3_dedup 37.80

| comparison | difference | 95% CI | P(> 0) | helped / hurt | sign test p |
|---|---|---|---|---|---|
| v3_dedup - video | -4.2 | [-10.8, +2.6] | 0.11 | 45 / 55 | 0.368 |
| v3_dedup - v3 | 1.7 | [-2.0, +5.3] | 0.81 | 15 / 8 | 0.210 |
| v3 - video | -5.8 | [-12.5, +1.0] | 0.05 | 41 / 58 | 0.107 |

## 2. Per task (descriptive)

| task | n | video | v3 | v3_dedup | dedup vs video helped/hurt | dedup vs v3 helped/hurt |
|---|---|---|---|---|---|---|
| obj_appearance_order | 25 | 48.0 | 44.0 | 44.0 | 3/4 | 0/0 |
| object_abs_distance | 25 | 43.2 | 31.2 | 42.4 | 13/9 | 7/2 |
| object_counting | 25 | 35.2 | 35.6 | 35.6 | 7/5 | 0/0 |
| object_rel_direction | 25 | 31.5 | 43.5 | 35.2 | 3/2 | 1/3 |
| object_rel_distance | 25 | 48.0 | 36.0 | 44.0 | 4/5 | 5/3 |
| object_size_estimation | 25 | 56.8 | 37.2 | 37.2 | 5/13 | 0/0 |
| room_size_estimation | 25 | 45.2 | 25.6 | 28.0 | 6/15 | 2/0 |
| route_planning | 25 | 28.0 | 36.0 | 36.0 | 4/2 | 0/0 |

## 3. Did dedup do its job? (75 relative-direction, relative-distance and absolute-distance questions; descriptive)

clean sheets (each named object exactly once): before 2 (3%), after 36 (48%); sheets changed by dedup 57; not clean after dedup because an object is missing: 39

(mean score x100; relative distance and direction are 0/1, absolute distance MRA)

| group | n | video | v3 | v3_dedup | dedup vs v3 helped/hurt | dedup vs video helped/hurt |
|---|---|---|---|---|---|---|
| clean after dedup, was duplicated (dedup changed the sheet) | 34 | 43.2 | 35.9 | 41.2 | 8/5 | 9/7 |
| clean before and after | 2 | 95.0 | 50.0 | 50.0 | 0/0 | 0/1 |
| not clean: an object missing | 39 | 36.4 | 37.4 | 40.0 | 5/3 | 11/8 |

answers that differ between v3 and v3_dedup: on changed sheets 26 of 57; on every other question 6 of 143

## 4. Relative direction (25 questions; descriptive)

|  | A | B | C | D |
|---|---|---|---|---|
| truth | 11 | 9 | 3 | 2 |
| video | 17 | 5 | 3 | 0 |
| v3 | 13 | 8 | 3 | 1 |
| v3_dedup | 13 | 9 | 2 | 1 |

questions whose dedup sheet has each named object once (one implied direction): 11 of 25; the implied direction is right in 10
- video: answers the sheet-implied direction in 3 of 11; right in 3
- v3: answers the sheet-implied direction in 5 of 11; right in 5
- v3_dedup: answers the sheet-implied direction in 3 of 11; right in 3

answers differing from video's: 5; of them video answered 'A' in 4; v3_dedup's new answer is the sheet-implied one in 0 of the 0 with one implied direction
answers differing from v3's: 4; v3_dedup's answer is the sheet-implied one in 0 of the 2 with one implied direction; v3's was in 2

## 5. Examples: v3_dedup against pipeline_v3, the first 5 helped and the first 5 hurt by question id (descriptive)

**helped**

- 604 (room_size_estimation): What is the size of this room (in square meters)? 
  options None; truth 25.2; video 30 (0.70); v3 11.99 (0.00); v3_dedup 18.02 (0.50); dedup sheet implies -; sheet changed by dedup: False
  before: 

- 680 (object_abs_distance): Measuring from the closest point of each object, what is the distance between the sofa and the stove (in meters)?
  options None; truth 2.9; video 2.5 (0.80); v3 1.376 (0.00); v3_dedup 3.1 (0.90); dedup sheet implies -; sheet changed by dedup: True
  before: sofa x3 [(-0.60,-0.35,1.38) m, 1 fr] [(-3.51,3.64,0.42) m, 2 fr] [(-3.45,2.26,0.09) m, 1 fr]; stove x0
  after:  sofa x1 [(-3.51,3.64,0.42) m, 2 fr]; stove x0

- 707 (object_abs_distance): Measuring from the closest point of each object, what is the distance between the washer and the refrigerator (in meters)?
  options None; truth 2.4; video 3.0 (0.50); v3 1.3 (0.10); v3_dedup 2.065 (0.80); dedup sheet implies -; sheet changed by dedup: True
  before: washer x2 [(0.09,-1.25,4.26) m, 4 fr] [(-0.80,-1.17,4.10) m, 3 fr]; refrigerator x2 [(-0.80,-1.16,4.11) m, 1 fr] [(-1.21,-0.61,2.05) m, 2 fr]
  after:  washer x1 [(0.09,-1.25,4.26) m, 4 fr]; refrigerator x1 [(-1.21,-0.61,2.05) m, 2 fr]

- 896 (object_abs_distance): Measuring from the closest point of each object, what is the distance between the table and the stove (in meters)?
  options None; truth 0.9; video 1.2 (0.40); v3 1.41 (0.00); v3_dedup 1.193 (0.40); dedup sheet implies -; sheet changed by dedup: True
  before: table x1 [(-0.18,-2.44,0.31) m, 2 fr]; stove x2 [(1.05,-3.21,0.47) m, 3 fr] [(1.47,-2.97,0.32) m, 1 fr]
  after:  table x1 [(-0.18,-2.44,0.31) m, 2 fr]; stove x1 [(1.05,-3.21,0.47) m, 3 fr]

- 907 (object_abs_distance): Measuring from the closest point of each object, what is the distance between the stove and the tv (in meters)?
  options None; truth 4.5; video 3.2 (0.50); v3 3.5 (0.60); v3_dedup 3.67 (0.70); dedup sheet implies -; sheet changed by dedup: False
  before: stove x1 [(0.43,-3.23,-0.44) m, 1 fr]; tv x0

**hurt**

- 798 (object_abs_distance): Measuring from the closest point of each object, what is the distance between the fireplace and the tv (in meters)?
  options None; truth 1.3; video 2.5 (0.00); v3 1.451 (0.80); v3_dedup 1.6 (0.60); dedup sheet implies -; sheet changed by dedup: True
  before: fireplace x2 [(1.39,-3.55,3.82) m, 5 fr] [(1.65,-3.25,4.10) m, 4 fr]; tv x1 [(-0.71,-3.83,3.75) m, 5 fr]
  after:  fireplace x1 [(1.39,-3.55,3.82) m, 5 fr]; tv x1 [(-0.71,-3.83,3.75) m, 5 fr]

- 1117 (object_rel_direction_medium): If I am standing by the refrigerator and facing the dishwasher, is the washer to my left, right, or back?
  options ['A. left', 'B. right', 'C. back']; truth B; video A (0.00); v3 B (1.00); v3_dedup A (0.00); dedup sheet implies B; sheet changed by dedup: True
  before: refrigerator x2 [(-0.80,-1.16,4.11) m, 1 fr] [(-1.21,-0.61,2.05) m, 2 fr]; dishwasher x2 [(-0.80,-1.25,4.05) m, 4 fr] [(0.12,-1.46,4.18) m, 2 fr]; washer x2 [(0.09,-1.24,4.26) m, 4 fr] [(-0.80,-1.17,4.11) m, 3 fr]
  after:  refrigerator x1 [(-1.21,-0.61,2.05) m, 2 fr]; dishwasher x1 [(-0.80,-1.25,4.05) m, 4 fr]; washer x1 [(0.09,-1.24,4.26) m, 4 fr]

- 1182 (object_rel_direction_medium): If I am standing by the table and facing the stool, is the tv to my left, right, or back?
  options ['A. right', 'B. back', 'C. left']; truth C; video A (0.00); v3 C (1.00); v3_dedup A (0.00); dedup sheet implies C; sheet changed by dedup: True
  before: table x4 [(-1.54,-0.65,2.44) m, 2 fr] [(-0.12,-1.65,3.94) m, 2 fr] [(0.16,-1.39,3.55) m, 1 fr] [(-1.49,-0.54,1.93) m, 2 fr]; stool x4 [(-1.53,-0.58,2.39) m, 3 fr] [(0.16,-1.39,3.55) m, 1 fr] [(-0.15,-1.67,3.94) m, 1 fr] [(-1.49,-0.42,2.12) m, 1 fr]; tv x1 [(0.15,-1.24,2.27) m, 2 fr]
  after:  table x1 [(-1.54,-0.65,2.44) m, 2 fr]; stool x1 [(-1.53,-0.58,2.39) m, 3 fr]; tv x1 [(0.15,-1.24,2.27) m, 2 fr]

- 1282 (object_rel_direction_easy): If I am standing by the table and facing the fireplace, is the stove to the left or the right of the fireplace?
  options ['A. right', 'B. left']; truth A; video A (1.00); v3 A (1.00); v3_dedup B (0.00); dedup sheet implies -; sheet changed by dedup: True
  before: table x3 [(-2.04,-1.82,2.68) m, 2 fr] [(-6.81,-2.26,2.95) m, 2 fr] [(-2.02,-1.15,1.87) m, 1 fr]; fireplace x1 [(-2.04,-3.40,3.37) m, 2 fr]; stove x0
  after:  table x1 [(-6.81,-2.26,2.95) m, 2 fr]; fireplace x1 [(-2.04,-3.40,3.37) m, 2 fr]; stove x0

- 1406 (object_rel_distance): Measuring from the closest point of each object, which of these objects (chair, table, tv, fireplace) is the closest to the sofa?
  options ['A. chair', 'B. table', 'C. tv', 'D. fireplace']; truth B; video A (0.00); v3 B (1.00); v3_dedup A (0.00); dedup sheet implies -; sheet changed by dedup: True
  before: chair x2 [(-0.93,-3.20,3.74) m, 4 fr] [(1.49,-0.51,1.56) m, 2 fr]; table x4 [(-0.21,-3.58,3.91) m, 4 fr] [(2.72,-1.74,2.23) m, 2 fr] [(1.02,-3.75,3.25) m, 1 fr] [(2.31,-3.79,3.17) m, 1 fr]; tv x1 [(-0.71,-3.83,3.75) m, 5 fr]; fireplace x2 [(1.39,-3.55,3.82) m, 5 fr] [(1.65,-3.25,4.10) m, 4 fr]; sofa x0
  after:  chair x1 [(-0.93,-3.20,3.74) m, 4 fr]; table x1 [(-0.21,-3.58,3.91) m, 4 fr]; tv x1 [(-0.71,-3.83,3.75) m, 5 fr]; fireplace x1 [(1.39,-3.55,3.82) m, 5 fr]; sofa x0
