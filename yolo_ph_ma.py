import cv2
import sys
import argparse
import numpy as np

from getBBs import yoloBBs

# Method parameters
MA_WINDOW = 5       # moving average window size
USE_ACC = False     # Whether accuracy should be used
CUP_ID = 41         # "cup" in yolo output
BALL_ID = 32        # "ball" in yolo output


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
    if not ret:
        print("Cannot read video file")
        sys.exit()

    # Initialize the YOLO object detecting
    bbGetter = yoloBBs()

    # Initialize
    bboxes = bbGetter.getBBs(frame)  # (x1, y1, x2, y2)
    NUM_CUPS = len(bboxes[bboxes[:, 5] == CUP_ID])
    prev_pos = np.column_stack([bboxes[:, 0] / 2 + bboxes[:, 2] / 2,
                                bboxes[:, 1] / 2 + bboxes[:, 3] / 2])
    prev_vel = np.zeros(prev_pos.shape)
    prev_acc = np.zeros(prev_pos.shape)

    poss += [prev_pos]
    vels += [prev_vel]
    accs += [prev_acc]

    # Update boxes from # (x1, y1, x2, y2) to (x1, y1, width, height)
    bboxes[:, 2] -= bboxes[:, 0]
    bboxes[:, 3] -= bboxes[:, 1]

    colors = [(np.random.randint(0, 255), np.random.randint(0, 255),
               np.random.randint(0, 255)) for x in range(len(bboxes))]

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

        # Visualize predictions and YOLO bounding boxes
        if visualize:
            cv2.putText(frame, "Data from Yolo and extrapolated", (100, 80),
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
                bboxes = tbboxes

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

        if yolo_found and len(poss) > 1:
            boxes = np.array(boxes).astype(float)
            # (x1, y1, width, height) to centroid
            pos = np.column_stack([boxes[:, 0] + boxes[:, 2] / 2,
                                   boxes[:, 1] + boxes[:, 3] / 2])

            # Update velocity and acceleration of the cups
            # from position differences
            vel = np.diff(poss, n=1, axis=0)[-MA_WINDOW:].mean(axis=0)
            acc = np.diff(poss, n=2, axis=0)[-MA_WINDOW:].mean(axis=0)

        # Append the values
        poss += [pos]
        vels += [vel]
        accs += [acc]

        # Display result
        if visualize:
            cv2.imshow("YOLO - PH + MA", frame)
            cv2.waitKey(1)

        prev_vel = vel
        prev_pos = pos
        prev_acc = acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="data/video01.mp4")

    args = parser.parse_args()
    run_tracking(args.video_path, True)
