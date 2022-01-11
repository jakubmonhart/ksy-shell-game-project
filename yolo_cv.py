import cv2
import numpy as np
import sys
import argparse
from yolov5.getBBs import yoloBBs 

def run_tracking(video_path, tracker_type):

  if tracker_type == "MOSSE":
    create_tracker = cv2.legacy.TrackerMOSSE_create
  elif tracker_type == "KCF":
    create_tracker = cv2.TrackerKCF_create

  video = cv2.VideoCapture(video_path)

  if (video.isOpened()== False):
    print("Error opening video file")
    sys.exit()

  # Read first frame.
  ret, frame = video.read()
  if ret==False:
    print("Cannot read video file")
    sys.exit()

  bbGetter = yoloBBs()

  bboxes = bbGetter.getBBs(frame)

  # Select boxes
  bboxes = [tuple(map(int, (x[0],x[1], (x[2]-x[0]), (x[3]-x[1])))) for x in bboxes]
  colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for x in range(len(bboxes))]

  print('Selected bounding boxes {}'.format(bboxes))

  # Init trackers
  trackers = []
  for bbox in bboxes:
    trackers.append(create_tracker())
    trackers[-1].init(frame, bbox)

  frame_skip=1
  t = 0
  while video.isOpened():
    t+=1

    # Read a new frame
    ret, frame = video.read()
    if ret==False:
       break 

    if (t%frame_skip != 0):
      continue
    
    # Start timer
    timer = cv2.getTickCount()

    # Update trackers
    boxes = []
    tracking_failure = False
    for tracker in trackers:
      ret, box = tracker.update(frame)
      
      if ret==False:
        tracking_failure = True

      boxes.append(box)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw tracked objects
    for i, box in enumerate(boxes):
      p1 = (int(box[0]), int(box[1]))
      p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
      cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    if tracking_failure:
      # Tracking failure
      cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    cv2.imshow("MultiTracking", frame)

    cv2.waitKey(1)

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--video_path", type=str ,default="data/video01.mp4")
  parser.add_argument("--tracker_type", type=str ,default="KCF")

  args = parser.parse_args()
  run_tracking(args.video_path, args.tracker_type)
