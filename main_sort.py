import cv2
import sys
import argparse
import numpy as np

from yolov5.getBBs import yoloBBs
from sort.sort import Sort

# SORT parameters
MAX_AGE = 10
MIN_HITS = 3
IOU_THRESHOLD = 0.2


def one_ball_three_cups(video_path: str, display: bool):

    # Try to load the video
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error opening video file")
        sys.exit()

    # Initialize YOLO object detection
    yolo = yoloBBs()

    # Initialize SORT motion tracking
    mot_tracker = Sort(MAX_AGE, MIN_HITS, IOU_THRESHOLD)

    colors = [(np.random.randint(0, 255), np.random.randint(0, 255),
               np.random.randint(0, 255)) for _ in range(10)]

    frame_skip = 1
    frame_no = 0
    while video.isOpened():
        frame_no += 1

        # Read a new frame
        ret, frame = video.read()
        if not ret:
            print("Cannot read video file")
            break

        if (frame_no % frame_skip != 0):
            continue

        # Get the bounding boxes in the form (x1, y1, x2, y2, score)
        bboxes = yolo.getBBs(frame)

        # Start timer
        timer = cv2.getTickCount()

        # Update SORT tracker
        if bboxes.any():
            trackers = mot_tracker.update(bboxes)
        else:
            trackers = mot_tracker.update(np.empty((0, 5)))

        # Visual output
        if(display):
            for d in trackers:
                d = d.astype(np.int32)
                cv2.rectangle(frame, (d[0], d[1]),
                              (d[2], d[3]), colors[int(d[4]) % 10], 2, 1)

            # Calculate Frames per second (FPS)
            fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

            # Display FPS on frame
            cv2.putText(frame, "FPS : " + str(int(fps)), (100, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50, 170, 50), 2)

            # Display result
            cv2.imshow("MultiTracking", frame)
            cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, default="/home/ondin/Dokumenty/CTU/7_term/KSY/ksy-shell-game-project/data/higher_slow.mp4")
    parser.add_argument('--no_display', dest='display', action='store_false',
                        help='Turn off online tracker output')

    args = parser.parse_args()
    one_ball_three_cups(args.video_path, args.display)
