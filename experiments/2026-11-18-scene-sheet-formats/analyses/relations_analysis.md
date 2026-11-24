## 1. Overall (dev200; dev results)

VSI-Bench overall: A_video 41.99, pipeline_v3 36.14, pipeline_v3_dedup 37.80, pipeline_v3_dedup_relations 44.17

| comparison | difference | 95% CI | P(> 0) | helped / hurt | sign test p |
|---|---|---|---|---|---|
| dedup_relations - video | 2.2 | [-5.2, +9.5] | 0.72 | 56 / 54 | 0.924 |
| dedup_relations - dedup | 6.4 | [+1.9, +10.8] | 1.00 | 24 / 10 | 0.024 |

## 2. Per task (descriptive)

| task | n | video | v3 | dedup | dedup_relations | rel. vs video helped/hurt | rel. vs dedup helped/hurt |
|---|---|---|---|---|---|---|---|
| object_abs_distance | 25 | 43.2 | 31.2 | 42.4 | 52.8 | 14/8 | 9/5 |
| object_rel_distance | 25 | 48.0 | 36.0 | 44.0 | 60.0 | 7/4 | 8/4 |
| object_rel_direction | 25 | 31.5 | 43.5 | 35.2 | 59.7 | 10/3 | 7/1 |
| obj_appearance_order | 25 | 48.0 | 44.0 | 44.0 | 44.0 | 3/4 | 0/0 |
| object_counting | 25 | 35.2 | 35.6 | 35.6 | 35.6 | 7/5 | 0/0 |
| object_size_estimation | 25 | 56.8 | 37.2 | 37.2 | 37.2 | 5/13 | 0/0 |
| room_size_estimation | 25 | 45.2 | 25.6 | 28.0 | 28.0 | 6/15 | 0/0 |
| route_planning | 25 | 28.0 | 36.0 | 36.0 | 36.0 | 4/2 | 0/0 |

## 3. Relative direction (25 questions; descriptive)

questions with a stated direction: 11; the stated direction is right in 10
- video: answers the stated direction in 3 of 11; right in 3 of 11; right on the other 14: 5
- v3: answers the stated direction in 5 of 11; right in 5 of 11; right on the other 14: 6
- dedup: answers the stated direction in 3 of 11; right in 3 of 11; right on the other 14: 6
- dedup_relations: answers the stated direction in 9 of 11; right in 9 of 11; right on the other 14: 6

answer letters, all 25:

|  | A | B | C | D |
|---|---|---|---|---|
| truth | 11 | 9 | 3 | 2 |
| video | 17 | 5 | 3 | 0 |
| v3 | 13 | 8 | 3 | 1 |
| dedup | 13 | 9 | 2 | 1 |
| dedup_relations | 10 | 10 | 3 | 2 |

on the 11 with a stated direction: truth {'A': 5, 'B': 4, 'C': 1, 'D': 1}, dedup_relations {'A': 3, 'B': 4, 'C': 3, 'D': 1}, video {'A': 6, 'B': 3, 'C': 2}

## 4. Distance questions (descriptive)

absolute distance: 12 of 25 questions get a precomputed distance. On those:
- precomputed vs truth: MRA 65.0, median ratio 0.89
- video: MRA 38.3; answer equals the precomputed distance (within 1%) in 0 of 12
- dedup: MRA 41.7; answer equals the precomputed distance (within 1%) in 0 of 12
- dedup_relations: MRA 65.0; answer equals the precomputed distance (within 1%) in 12 of 12
- on the 13 without one: video 47.7, dedup_relations 41.5

relative distance: 19 of 25 get a precomputed closest object; it is right in 13
- video: answers the precomputed object in 8 of 19; right in 9 of 19
- dedup: answers the precomputed object in 8 of 19; right in 8 of 19
- dedup_relations: answers the precomputed object in 19 of 19; right in 13 of 19

## 5. Question 1282 (tie-break)

If I am standing by the table and facing the fireplace, is the stove to the left or the right of the fireplace?
options ['A. right', 'B. left']; truth A; video A; v3 A; dedup B; dedup_relations B
dedup sheet: [('table', -6.81, 2.95, '2'), ('fireplace', -2.04, 3.37, '2')]
dedup_relations sheet: [('table', -6.81, 2.95, '2'), ('fireplace', -2.04, 3.37, '2')]
relation line: (none: an object is missing)

## 6. Simulated full method (CHOSEN ON DEV: not a confirmed result)

dedup_relations for distance and direction questions, video for object and room size, dedup for the rest
overall 48.77

| comparison | difference | 95% CI | P(> 0) | helped / hurt | sign test p |
|---|---|---|---|---|---|
| full method - video | 6.8 | [+0.1, +13.4] | 0.98 | 45 / 26 | 0.032 |
| full method - dedup | 11.0 | [+5.5, +16.5] | 1.00 | 52 / 21 | 0.000 |

## 7. Examples: dedup_relations against dedup, the first 5 helped and the first 5 hurt by question id

**helped**
- 798 (object_abs_distance): Measuring from the closest point of each object, what is the distance between the fireplace and the tv (in meters)?
  truth 1.3; video 2.5 (0.00); dedup 1.6 (0.60); dedup_relations 1.043 (0.70)
  named rows (metres): [('fireplace', '1', 1.39, 3.82), ('tv', '1', -0.71, 3.75)]
  relation: Closest-point distance between the fireplace and the tv: 1.043 m.
- 896 (object_abs_distance): Measuring from the closest point of each object, what is the distance between the table and the stove (in meters)?
  truth 0.9; video 1.2 (0.40); dedup 1.193 (0.40); dedup_relations 0.683 (0.60)
  named rows (metres): [('table', '1', -0.18, 0.31), ('stove', '1', 1.05, 0.47)]
  relation: Closest-point distance between the table and the stove: 0.683 m.
- 1117 (object_rel_direction_medium): If I am standing by the refrigerator and facing the dishwasher, is the washer to my left, right, or back?
  truth B; video A (0.00); dedup A (0.00); dedup_relations B (1.00)
  named rows (metres): [('refrigerator', '2', -1.21, 2.05), ('dishwasher', '1', -0.8, 4.05), ('washer', '1', 0.09, 4.26)]
  relation: Direction from the sheet's positions (the floor seen from above): standing by the refrigerator and facing the dishwasher, the washer is to your right.
- 1182 (object_rel_direction_medium): If I am standing by the table and facing the stool, is the tv to my left, right, or back?
  truth C; video A (0.00); dedup A (0.00); dedup_relations C (1.00)
  named rows (metres): [('table', '1', -1.54, 2.44), ('stool', '1', -1.53, 2.39), ('tv', '1', 0.15, 2.27)]
  relation: Direction from the sheet's positions (the floor seen from above): standing by the table and facing the stool, the tv is to your left.
- 1247 (object_rel_direction_easy): If I am standing by the dishwasher and facing the refrigerator, is the washer to the left or the right of the refrigerator?
  truth B; video A (0.00); dedup A (0.00); dedup_relations B (1.00)
  named rows (metres): [('dishwasher', '1', -0.8, 4.05), ('refrigerator', '2', -1.21, 2.05), ('washer', '1', 0.09, 4.26)]
  relation: Direction from the sheet's positions (the floor seen from above): standing by the dishwasher and facing the refrigerator, the washer is to the left of the refrigerator.

**hurt**
- 707 (object_abs_distance): Measuring from the closest point of each object, what is the distance between the washer and the refrigerator (in meters)?
  truth 2.4; video 3.0 (0.50); dedup 2.065 (0.80); dedup_relations 1.729 (0.50)
  named rows (metres): [('washer', '1', 0.09, 4.26), ('refrigerator', '2', -1.21, 2.05)]
  relation: Closest-point distance between the washer and the refrigerator: 1.729 m.
- 907 (object_abs_distance): Measuring from the closest point of each object, what is the distance between the stove and the tv (in meters)?
  truth 4.5; video 3.2 (0.50); dedup 3.67 (0.70); dedup_relations 3.5 (0.60)
  named rows (metres): [('stove', '1', 0.43, -0.44)]
  relation: (none)
- 912 (object_abs_distance): Measuring from the closest point of each object, what is the distance between the bathtub and the tv (in meters)?
  truth 3.5; video 2.5 (0.50); dedup 2.3 (0.40); dedup_relations 2.1 (0.30)
  named rows (metres): [('tv', '1', 1.09, -0.62)]
  relation: (none)
- 1918 (object_rel_distance): Measuring from the closest point of each object, which of these objects (computer tower, door, trash can, ceiling light) is the closest to the telepho
  truth A; video A (1.00); dedup A (1.00); dedup_relations C (0.00)
  named rows (metres): [('computer tower', '1', 2.04, -1.5), ('door', '1', -0.35, 2.13), ('trash can', '1', -0.55, 0.83)]
  relation: door_1,trash_can_1,0.885
- 3005 (object_rel_distance): Measuring from the closest point of each object, which of these objects (telephone, bed, trash bin, table) is the closest to the towel?
  truth C; video D (0.00); dedup C (1.00); dedup_relations B (0.00)
  named rows (metres): [('telephone', '1', -0.58, -2.2), ('bed', '1', -0.13, 1.21), ('trash bin', '2', -1.4, -0.74), ('table', '5', -0.83, -1.83), ('towel', '3', -1.87, 1.75)]
  relation: Distance from each candidate to the nearest towel (closest points, metres): telephone 3.684; bed 0.695; trash bin 2.230; table 3.027. Closest to the towel: bed.

