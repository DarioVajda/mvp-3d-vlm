**1. object_counting** ([`scannetpp/bde1e479ad.mp4`](data/raw/VSI-Bench/scannetpp/bde1e479ad.mp4), id 2556)

> How many ceiling light(s) are in this room?

GT **8** · **4B: `2` (MRA 0.0)** · 2B: `4` (MRA 0.1)

![](results/examples/vsi_2556.jpg)

**2. object_abs_distance** ([`scannetpp/27dd4da69e.mp4`](data/raw/VSI-Bench/scannetpp/27dd4da69e.mp4), id 1794)

> Measuring from the closest point of each object, what is the distance between the basket and the cutting board (in meters)?

GT **9.1** · **4B: `2.3` (MRA 0.0)** · 2B: `2.0` (MRA 0.0)

![](results/examples/vsi_1794.jpg)

**3. object_size_estimation** ([`scannetpp/286b55a2bf.mp4`](data/raw/VSI-Bench/scannetpp/286b55a2bf.mp4), id 2288)

> What is the length of the longest dimension (length, width, or height) of the exhaust fan, measured in centimeters?

GT **12** · **4B: `25` (MRA 0.0)** · 2B: `24` (MRA 0.0)

![](results/examples/vsi_2288.jpg)

**4. room_size_estimation** ([`scannetpp/3f15a9266d.mp4`](data/raw/VSI-Bench/scannetpp/3f15a9266d.mp4), id 2672)

> What is the size of this room (in square meters)? 
> If multiple rooms are shown, estimate the size of the combined space.

GT **11.9** · **4B: `18.0` (MRA 0.0)** · 2B: `20.0` (MRA 0.0)

![](results/examples/vsi_2672.jpg)

**5. object_rel_distance** ([`scannetpp/21d970d8de.mp4`](data/raw/VSI-Bench/scannetpp/21d970d8de.mp4), id 1988)

> Measuring from the closest point of each object, which of these objects (tv, whiteboard, telephone, chair) is the closest to the door?

A. tv · B. whiteboard · C. telephone · D. chair

GT **B** · **4B: `D. chair` ✗** · 2B: `B. whiteboard` ✓

![](results/examples/vsi_1988.jpg)

**6. object_rel_direction_easy** ([`scannet/scene0357_01.mp4`](data/raw/VSI-Bench/scannet/scene0357_01.mp4), id 3710)

> If I am standing by the window and facing the towel, is the table to the left or the right of the towel?

A. left · B. right

GT **B** · **4B: `A. left` ✗** · 2B: `A. left` ✗

![](results/examples/vsi_3710.jpg)

**7. route_planning** ([`arkitscenes/47331262.mp4`](data/raw/VSI-Bench/arkitscenes/47331262.mp4), id 5073)

> You are a robot beginning at the corner of the table towards the sofa and facing the table. You want to navigate to the corner fan. You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'): 1. [please fill in] 2. Go forward until the bookshelf 3. [please fill in] 4. Go forward until the wall. You have reached the final destination.

A. Turn Back, Turn Right · B. Turn Right, Turn Left · C. Turn Left, Turn Right · D. Turn Right, Turn Right

GT **C** · **4B: `D. Turn Right, Turn Right` ✗** · 2B: `B. Turn Right, Turn Left` ✗

![](results/examples/vsi_5073.jpg)

**8. obj_appearance_order** ([`scannet/scene0580_01.mp4`](data/raw/VSI-Bench/scannet/scene0580_01.mp4), id 4069)

> What will be the first-time appearance order of the following categories in the video: door, table, chair, bookshelf?

A. chair, bookshelf, door, table · B. bookshelf, table, chair, door · C. door, table, chair, bookshelf · D. door, chair, table, bookshelf

GT **D** · **4B: `C. door, table, chair, bookshelf` ✗** · 2B: `D. door, chair, table, bookshelf` ✓

![](results/examples/vsi_4069.jpg)

**9. object_counting** ([`scannetpp/21d970d8de.mp4`](data/raw/VSI-Bench/scannetpp/21d970d8de.mp4), id 2546)

> How many ceiling light(s) are in this room?

GT **12** · **4B: `4` (MRA 0.0)** · 2B: `4` (MRA 0.0)

![](results/examples/vsi_2546.jpg)

**10. object_abs_distance** ([`arkitscenes/42897629.mp4`](data/raw/VSI-Bench/arkitscenes/42897629.mp4), id 751)

> Measuring from the closest point of each object, what is the distance between the tv and the refrigerator (in meters)?

GT **9.1** · **4B: `1.0` (MRA 0.0)** · 2B: `4.4` (MRA 0.0)

![](results/examples/vsi_751.jpg)
