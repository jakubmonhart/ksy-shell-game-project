import cv2
import sys
import argparse
import numpy as np

from getBBs import yoloBBs

# Method parameters
ALPHA_V = 0.7       # EMA constant for VEL
ALPHA_A = ALPHA_V   # EMA constant for ACC
CUP_ID = 41         # "cup" in yolo output
BALL_ID = 32        # "ball" in yolo output
USE_ACC = False     # Whether acceleration should be used

# Minimal number of frames between start and end
FINAL_BALL_SAFE_TIMEZONE = 30
# Number of frames in which the ball position is evaluated
FINAL_BALL_EVALUATION_ZONE = 10


def run_tracking(video_path: str, visualize: bool = False):
    # Value history
    vels = []
    poss = []
    accs = []

    # Load video and ensure it was opened correctly
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print("Error opening video file")
        sys.exit()

    # Read first frame
    ret, frame = video.read()
    init_frame = np.copy(frame)
    if not ret:
        print("Cannot read video file")
        sys.exit()

    # Initialize the YOLO object detecting
    bbGetter = yoloBBs()

    # Initialize
    bboxes = bbGetter.getBBs(frame)
    NUM_CUPS = len(bboxes[bboxes[:, 5] == CUP_ID])
    prev_pos = np.column_stack([bboxes[:, 0] / 2 + bboxes[:, 2] / 2,
                                bboxes[:, 1] / 2 + bboxes[:, 3] / 2])
    prev_vel = np.zeros(prev_pos.shape)
    prev_acc = np.zeros(prev_pos.shape)

    poss += [prev_pos]
    vels += [prev_vel]
    accs += [prev_acc]

    # Select boxes
    bboxes[:, 2] -= bboxes[:, 0]
    bboxes[:, 3] -= bboxes[:, 1]
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255),
               np.random.randint(0, 255)) for x in range(len(bboxes))]

    
    # Variables for evaluating of results
    init_cup = None
    fin_ball = None
    init_cup_frame = 0
    fin_cup_frame = 0

    # Read the video frame by frame
    while video.isOpened():

        # Read a new frame
        ret, frame = video.read()
        if not ret:
            break

        # Update physical prediction
        pos = prev_pos + prev_vel
        if USE_ACC:
            vel = prev_vel + prev_acc
        else:
            vel = prev_vel
        acc = prev_acc

        # Get boudning boxes using YOLO
        tbboxes = bbGetter.getBBs(frame)
        
        # Contains ball
        if tbboxes.any() and len(tbboxes[tbboxes[:, 5] == BALL_ID]) == 1:
            ball_bb = tbboxes[tbboxes[:, 5] == BALL_ID][0]

            # Haven't found intial cup yet
            if init_cup is None:
                tposs = np.array(poss)
                # Select the cup closest to the ball
                # (use average over multiple frames to avoid glitches)
                init_cup = np.argmin(abs(tposs[:, :, 0].mean(axis=0) - (ball_bb[2] + ball_bb[0]) / 2))
                init_cup_frame = len(poss) - 1

            # found initial cup and "safe zone" passed
            elif len(poss) + 1 - init_cup_frame > FINAL_BALL_SAFE_TIMEZONE:
                fin_ball = np.copy(ball_bb)
                fin_cup_frame = len(poss) - 1

        if visualize:
            cv2.putText(frame, "Data from Yolo and extrapoladed", (100, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

            for i in range(NUM_CUPS):
                x_s = bboxes[i][2]
                y_s = bboxes[i][3]
                px1 = (int(pos[i][0] - x_s / 2), int(pos[i][1] - y_s / 2))
                px2 = (int(pos[i][0] + x_s / 2), int(pos[i][1] + y_s / 2))
                cv2.rectangle(frame, px1, px2, colors[i], 2, 1)

            for i, box in enumerate(tbboxes):
                p1 = (int(box[0]), int(box[1]))
                p2 = (int(box[2]), int(box[3]))
                cv2.rectangle(frame, p1, p2, [0, 0, 0], 2, 1)

        yolo_found = False
        # Check whether yolo found all cups
        if tbboxes.any() and len(tbboxes[tbboxes[:, 5] == CUP_ID]) == NUM_CUPS:
            # Calculate centroid of the bounding box found by YOLO
            yolo_pos = np.column_stack([tbboxes[:, 0] / 2 + tbboxes[:, 2] / 2,
                                        tbboxes[:, 1] / 2 + tbboxes[:, 3] / 2])

            # Calculate distance between yolo detections
            # - this is used to filter out frames where yolo detects
            #   the same cup twice
            D = np.zeros((NUM_CUPS, NUM_CUPS), dtype="float32")
            for i in range(NUM_CUPS):
                for j in range(NUM_CUPS):
                    D[i, j] = np.linalg.norm(yolo_pos[i] - yolo_pos[j])
            threshold = 10
            if (D + np.eye(NUM_CUPS) * 2 * threshold < threshold).any():
                # YOLO detected one object twice
                yolo_found = False
            else:
                # All detections are unique
                yolo_found = True
                bboxes = tbboxes[tbboxes[:, 5] == CUP_ID]

                # Update boxes from # (x1, y1, x2, y2) to (x1, y1, width, height)
                bboxes[:, 2] -= bboxes[:, 0]
                bboxes[:, 3] -= bboxes[:, 1]

                # Match the new detected boxes to the previous ones
                C = np.zeros((NUM_CUPS, NUM_CUPS), dtype="float32")
                for i in range(NUM_CUPS):
                    for j in range(NUM_CUPS):
                        C[i, j] = np.linalg.norm(yolo_pos[i] - pos[j])

                matched = np.zeros((2, NUM_CUPS), dtype=int)
                boxes = np.zeros([NUM_CUPS, 4])

                # Get best unique matches (do not match two predictions to one cup)
                for (a, b) in zip(*np.unravel_index(np.argsort(C, axis=None), C.shape)):
                    if matched[0, a] or matched[1, b]:
                        continue
                    matched[0, a] = 1
                    matched[1, b] = 1
                    boxes[b] = bboxes[a][0:4]

        if yolo_found:
            boxes = np.array(boxes).astype(float)
            # (x1, y1, width, height) to centroid
            pos = np.column_stack([boxes[:, 0] + boxes[:, 2] / 2,
                                   boxes[:, 1] + boxes[:, 3] / 2])

            # Update velocity and acceleration of the cups
            # from position differences
            vel = (pos - prev_pos) * ALPHA_V + (1 - ALPHA_V) * prev_vel
            acc = (vel - prev_vel) * ALPHA_A + (1 - ALPHA_A) * prev_acc

        # Append the values
        poss += [pos]
        vels += [vel]
        accs += [acc]

        # Display result
        if visualize:
            cv2.imshow("YOLO - PH + EMA", frame)
            cv2.waitKey(1)

        prev_vel = vel
        prev_pos = pos
        prev_acc = acc
    
    # evaluate results

    ret = []
    if init_cup is None or fin_ball is None:  # no ball found, cannot evaluate results
        if visualize:
            cv2.putText(init_frame, f"Ball not detected!", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255) ,2)
        ret = (0, 0)
    else:
        tposs = np.array(poss)
        fin_cup = np.argmin(abs(tposs[fin_cup_frame-FINAL_BALL_EVALUATION_ZONE:,:,0].mean(axis=0) - (fin_ball[2] + fin_ball[0])/2))
        if visualize:
            cv2.putText(init_frame, f"Ball started under cup {init_cup}.", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[init_cup],2)
            cv2.putText(init_frame, f"And ended under cup {fin_cup}.", (100,110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[fin_cup],2)
    
    if visualize:
        cv2.imshow("MultiTracking", init_frame)
        cv2.waitKey(3000)

    if fin_cup == init_cup:
        ret = (1,1)
    else:
        ret= (1,0)

    return ret

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="data/video01.mp4")
    parser.add_argument("--show_vid", action="store_true")

    args = parser.parse_args()
    run_tracking(args.video_path, args.show_vid)
