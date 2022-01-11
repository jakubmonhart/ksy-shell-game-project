import cv2
import numpy as np
import sys
import argparse
import matplotlib.pyplot as plt
from yolov5.getBBs import yoloBBs
import numpy as np 

ALPHA_V = 0.8
ALPHA_A = ALPHA_V
CUP_ID = 41
BALL_ID = 32
WIN_SIZE = 5

def run_tracking(video_path, tracker_type):
  vels = []
  poss = []  
  accs = []
 
  video = cv2.VideoCapture(video_path)

  if (video.isOpened()== False):
    print("Error opening video file")
    sys.exit()

  # Read until frame.
  ret, frame = video.read()
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

  yolo = False
  frame_skip = 1
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

    pos = prev_pos + prev_vel
    vel = prev_vel #+ prev_acc
    acc = prev_acc

    tbboxes = bbGetter.getBBs(frame)
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
   

    yoled = False
    if tbboxes.any() and len(tbboxes[tbboxes[:,5]==CUP_ID]) == NUM_CUPS:
      yoled = True
      yolo_pos = np.column_stack([tbboxes[:,0]/2 + tbboxes[:,2]/2,tbboxes[:,1]/2 + tbboxes[:,3]/2])

      D = np.zeros((NUM_CUPS, NUM_CUPS), dtype="float32")
      for i in range(NUM_CUPS):
        for j in range(NUM_CUPS):
          D[i, j] = np.linalg.norm(yolo_pos[i]-yolo_pos[j])
      threshold = 10
      if (D+np.eye(NUM_CUPS)*2*threshold < threshold).any():
        yoled = False
      else:
        bboxes = tbboxes
        bboxes[:,2] -= bboxes[:,0]
        bboxes[:,3] -= bboxes[:,1]
        C = np.zeros((NUM_CUPS, NUM_CUPS), dtype="float32")
        for i in range(NUM_CUPS):
          for j in range(NUM_CUPS):
            C[i, j] = np.linalg.norm(yolo_pos[i]-pos[j])
        
        matched = np.zeros((2, NUM_CUPS), dtype=int)
        boxes = np.zeros([NUM_CUPS,4])
        # Get best unique matches (do not match two predictions to one cup)
        for (a, b) in zip(*np.unravel_index(np.argsort(C, axis=None), C.shape)):
          if matched[0, a] or matched[1, b]:
            continue
          matched[0, a] = 1
          matched[1, b] = 1
          boxes[b] = bboxes[a][0:4]

    if yoled and len(poss) > 1:
      boxes = np.array(boxes).astype(float)
      pos = np.column_stack([boxes[:,0] + boxes[:,2]/2,boxes[:,1] + boxes[:,3]/2])
      vel = np.diff(poss, axis=0)[-WIN_SIZE:].mean(axis=0) # prev_vel*0.05 + 0.95*(pos - prev_pos)
      acc = (vel - prev_vel)*ALPHA_A + (1-ALPHA_A)*prev_acc
    
    poss += [pos]
    vels += [vel]
    accs += [acc]


    # Display result
    cv2.imshow("MultiTracking", frame)

    cv2.waitKey(1)
    prev_vel = vel
    prev_pos = pos
    prev_acc = acc
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--video_path", type=str ,default="data/video01.mp4")
  parser.add_argument("--tracker_type", type=str ,default="KCF")

  args = parser.parse_args()
  run_tracking(args.video_path, args.tracker_type)
