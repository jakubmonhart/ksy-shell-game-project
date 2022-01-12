import cv2
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from getBBs import yoloBBs
import numpy as np 

ALPHA_V = 0.8
ALPHA_A = ALPHA_V
CUP_ID = 41
BALL_ID = 32

FINAL_BALL_SAFE_TIMEZONE = 30
FINAL_BALL_EVALUATION_ZONE = 10

def run_tracking(video_path, tracker_type):
  vels = []
  poss = []  
  accs = []

  if tracker_type == "MOSSE":
    create_tracker = cv2.legacy.TrackerMOSSE_create
  elif tracker_type == "KCF":  
    create_tracker = cv2.TrackerKCF_create

  video = cv2.VideoCapture(video_path)

  if (video.isOpened()== False):
    print("Error opening video file")
    sys.exit()

  # Read until frame.
  ret, frame = video.read()
  init_frame = np.copy(frame)
  if ret==False:
    print("Cannot read video file")
    sys.exit()

  bbGetter = yoloBBs()

  bboxes = bbGetter.getBBs(frame)
  NUM_CUPS = len(bboxes[bboxes[:,5]==CUP_ID])
  prev_pos = np.column_stack([bboxes[:,0]/2 + bboxes[:,2]/2,bboxes[:,1]/2 + bboxes[:,3]/2])
  prev_vel = np.zeros(prev_pos.shape)
  prev_acc = np.zeros(prev_pos.shape)
  poss += [prev_pos]
  vels += [prev_vel]
  accs += [prev_acc]

  # Select boxes
  bboxes[:,2] -= bboxes[:,0]
  bboxes[:,3] -= bboxes[:,1]
  colors = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for x in range(len(bboxes))]
  print('Selected bounding boxes {}'.format(bboxes))

  # Init trackers
  trackers = []
  for bbox in bboxes:
    trackers.append(create_tracker())
    trackers[-1].init(frame, bbox[:4].astype(int))

  tracking = True
  yolo = False
  frame_skip = 1
  t = 0

  # variables for results evaluating
  init_cup = None
  fin_ball = None
  init_cup_frame = 0
  fin_cup_frame = 0

  while video.isOpened():
    t+=1

    # Read a new frame
    ret, frame = video.read()
    if ret==False:
       break 

    if (t%frame_skip != 0):
      continue

    tbboxes = bbGetter.getBBs(frame)
    if tbboxes.any() and len(tbboxes[tbboxes[:,5]==BALL_ID])==1: #contains on ball
      ball_bb = tbboxes[tbboxes[:,5]==BALL_ID][0] 
      if init_cup is None: # not found intial cup yet
        tposs = np.array(poss)
        init_cup = np.argmin(abs(tposs[:,:,0].mean(axis=0) - (ball_bb[2] + ball_bb[0])/2))
        init_cup_frame = len(poss)-1
      elif len(poss)+1 - init_cup_frame > FINAL_BALL_SAFE_TIMEZONE: # found initial cup and "safe zone" passed
        fin_ball = np.copy(ball_bb)
        fin_cup_frame = len(poss)-1
        
    
    # Start timer
    timer = cv2.getTickCount()

    # Update trackers
    boxes = []
    if tracking:
      tracking_failure = False
      for tracker in trackers:
        ret, box = tracker.update(frame)

        if ret==False:
          tracking_failure = True

        boxes.append(box)
    
      if tracking_failure:
          tracking = False


    pos = prev_pos + prev_vel
    vel = prev_vel #+ prev_acc
    acc = prev_acc
    
    # Draw tracked objects
    if tracking:
      for i, box in enumerate(boxes):
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
        cv2.rectangle(frame, p1, p2, colors[i], 2, 1)

    if not tracking:
      #tbboxes = bbGetter.getBBs(frame)
      cv2.putText(frame, "Data from Yolo and extrapoladed", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
      for i in range(NUM_CUPS):
        x_s = bboxes[i][2]
        y_s = bboxes[i][3]
        px1 = (int(pos[i][0] - x_s/2), int(pos[i][1]-y_s/2))
        px2 = (int(pos[i][0] + x_s/2), int(pos[i][1]+y_s/2))
        cv2.rectangle(frame, px1, px2, colors[i], 2, 1)

      for i, box in enumerate(tbboxes):
        
        p1 = (int(box[0]), int(box[1]))
        p2 = (int(box[2]), int(box[3]))
        cv2.rectangle(frame, p1, p2, [0,0,0], 2, 1)
        cv2.imshow("MultiTracking", frame)

      if tbboxes.any() and len(tbboxes[tbboxes[:,5]==CUP_ID]) == NUM_CUPS:
        bboxes = tbboxes[tbboxes[:,5]==CUP_ID]
        yolo_pos = np.column_stack([bboxes[:,0]/2 + bboxes[:,2]/2,bboxes[:,1]/2 + bboxes[:,3]/2])
        bboxes[:,2] -= bboxes[:,0]
        bboxes[:,3] -= bboxes[:,1]
        C = np.zeros((NUM_CUPS, NUM_CUPS), dtype="float32")
        for i in range(NUM_CUPS):
          for j in range(NUM_CUPS):
            C[i, j] = np.linalg.norm(yolo_pos[i]-pos[j])
        
        matched = np.zeros((2, NUM_CUPS), dtype=int)
        boxes = np.zeros([NUM_CUPS, 4])
        # Get three unique matches (do not match two predictions to one cup)
        for (a, b) in zip(*np.unravel_index(np.argsort(C, axis=None), C.shape)):
          if matched[0, a] or matched[1, b]:
            continue
          matched[0, a] = 1
          matched[1, b] = 1

          trackers[b] = create_tracker()
          trackers[b].init(frame, bboxes[a][0:4].astype(int))
          boxes[b] = bboxes[a][0:4]
        tracking = True

    if tracking:
      boxes = np.array(boxes).astype(float)
      pos = np.column_stack([boxes[:,0] + boxes[:,2]/2,boxes[:,1] + boxes[:,3]/2])
      vel = (pos - prev_pos)*ALPHA_V + (1-ALPHA_V)*prev_vel # prev_vel*0.05 + 0.95*(pos - prev_pos)
      acc = (vel - prev_vel)*ALPHA_A + (1-ALPHA_A)*prev_acc
    
    poss += [pos]
    vels += [vel]
    accs += [acc]

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    if tracking_failure:
      # Tracking failure
      cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
    if not tracking:
      # Tracking failure
      cv2.putText(frame, "Data from physical model", (100,120), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
    cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);

    # Display FPS on frame
    cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    cv2.imshow("MultiTracking", frame)

    cv2.waitKey(1)
    prev_vel = vel
    prev_pos = pos
    prev_acc = acc
  
  # evaluate results
  ret = []
  if init_cup is None or fin_ball is None:  # no ball found, cannot evaluate results
    cv2.putText(init_frame, f"Ball not detected!", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255) ,2)
    ret = (0, 0)
  else:
    tposs = np.array(poss)
    fin_cup = np.argmin(abs(tposs[fin_cup_frame-FINAL_BALL_EVALUATION_ZONE:,:,0].mean(axis=0) - (fin_ball[2] + fin_ball[0])/2))
    cv2.putText(init_frame, f"Ball started under cup {init_cup}.", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[init_cup],2)
    cv2.putText(init_frame, f"And ended under cup {fin_cup}.", (100,110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, colors[fin_cup],2)
  
  cv2.imshow("MultiTracking", init_frame)

  cv2.waitKey(3000)

  if fin_cup == init_cup:
    ret = (1,1)
  else:
    ret= (1,0)

  return ret


  
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--video_path", type=str ,default="data/video01.mp4")
  parser.add_argument("--tracker_type", type=str ,default="KCF")

  args = parser.parse_args()
  run_tracking(args.video_path, args.tracker_type)
