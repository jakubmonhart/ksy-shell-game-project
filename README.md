# Assignment
**Vytvořte set videí (reálných/simulace) se hrou skořápky. Poté najděte algoritmus, který hru z videí dokáže vyřešit (uhodnout, kde je schovaný míček)**
Úloha zaměřená na pozornost. Vytvořte si set např. 10 videí, ve kterých je prezentována    hra skořápky s různou obtížností (danou délkou míchání). Poté zkuste vytvořit/natrénovat algoritmus, který dokáže z videa odhadnout, pod kterou skořápkou (kalíškem, nádobkou) se skrývá hledaný objekt. Konkrétní forma hry je na vás, může s kalíšky hýbat člověk nebo se mohou hýbat samy. Barvy, rychlost pohybu atd. jsou libovolné. Jediná podmínka je, že všechny tři kalíšky musí být stejné. Úspěšnost algoritmu ověřte na všech videích.  Zamyslete se, jak tuto úlohu zpracovává lidský mozek a v čem se tento mechanismus liší od vašeho řešení. 
Odkazy: https://cs.wikipedia.org/wiki/Sko%C5%99%C3%A1pky, https://cw.fel.cvut.cz/b201/_media/courses/a6m33ksy/prezentace3_2014ks.pdf
Doporučené nástroje, knihovny apod.: OpenCV, Tensorflow

# Installation
1. Clone the repository recursively:
`git clone --recurse-submodules https://github.com/jakubmonhart/ksy-shell-game-project`

If you already cloned and forgot to use `--recurse-submodules` you can run `git submodule update --init`

2. Make sure that you fulfill all the requirements: Python 3.8 or later with all requirements.txt dependencies installed.
To install, run:
`pip install -r requirements.txt`

# Dataset
Download dataset to `data/` using https://owncloud.cesnet.cz/index.php/s/rhwg0rC2Bso8dL3.

# Usage
YoloV5 + Deep Sort:
`python3 deep_sort_track.py --source ../temporary/ksy-shell-game-project/data/001_higher_slow_3_0.mp4 --show-vid`

# _**DEPRECATED**_

## Object tracking process
- Detect objects.
- Track detected objects.
- If the tracker looses track of some object, run the detection again and track from now on.
- Each detected and tracked object has unique id - need to keep track of that.

## What is implemented now
- Tracking of cups given initial bounding boxes. (bboxes are selected manually using opencv gui).
- Tracking with MOSSE and KCF

## Roadmap
0. Speed up the tracking by multithreading/multiprocessing?
1. Implement detection of cups. All cups are of same color on white background - use method for poles detection implemented in (https://github.com/m-minarik/lar-projekt).
2. Init the tracking process by automatically detecting the cups - pass detected bounding boxes to initialize one tracker for each cup.
3. If a tracker loses track of its cup, run the detection from step 2 and continue tracking (assign correct id to newly detected cup).
	- We can keep movement history of each cup. Then use this information to correctly assign id to lost cup. (compute velocity of cup and predict its future position?)
4. Add ball to the game. At the beginning of video, detect which cup hides the ball. Automatically evaluate if our algorithm correctly predicted ball position at the end of the video.
5.  Complete the dataset, evaluate our alogrithm.
6.  Prepare some slides for presentation.

## Tracking methods
**KCF**
- Slow (\~5 fps for 3 cups) but more accurate.
- Tried frame skipping, but tracking fails (due to shaking camera?).
- When KCF loses track of some cup, it stops tracking it altogether.

**MOSSE**
- Fast (\~30 fps for 3 cups) but less accurate.
- When MOSSE loses track of some cup, it predicts its position as the last seen one.

**Other methods**
- Try SORT (https://github.com/abewley/sort ,https://towardsdatascience.com/object-detection-and-tracking-in-pytorch-b3cf1a696a98).
- Try DeepSORT (https://github.com/nwojke/deep_sort)?
- Try GOTURN - implemented in opencv, but requires some model files (https://docs.opencv.org/3.4/d7/d4c/classcv_1_1TrackerGOTURN.html).


## Another approach
- Following repositories maybe implement the process above. Some of the following approaches use NN.
https://www.reddit.com/r/computervision/comments/rd1zbs/comment/ho015i6/?utm_source=share&utm_medium=web2x&context=3
- https://github.com/PaddlePaddle/PaddleDetection
- https://github.com/visionml/pytracking
- https://paperswithcode.com/task/object-tracking
- https://paperswithcode.com/task/person-re-identification
