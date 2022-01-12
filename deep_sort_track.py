# limit the number of cpus used by high performance libraries
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov5')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn

from yolov5.models.experimental import attempt_load
from yolov5.utils.downloads import attempt_download
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, 
                  check_imshow, xyxy2xywh, increment_path)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 deepsort root directory
if str(ROOT) not in sys.path:
  sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


##### CUP DETECTION #####
CUP_CLASS = 41
BALL_CLASS = 32


class Cup:
  def __init__(self, cup_id: int, pos: list):
    self.id = cup_id
    self.positions = [pos]

  def update(self, pos: list, cup_id=None):
    self.positions.append(pos)

    if cup_id:
      self.id = cup_id

class CupDetector:
  def __init__(self):
    self.cups = {}
    self.ball_start = None
    self.ball_end = None
    self.ball_count = 0
    self.cup_with_ball = -1
    self.cup_with_ball_true = -1
    self.message = None
    self.message_count = 0
    self.prediction_made = False
    self.correct = 0

  def add_cup(self, cup: Cup):
    self.cups[cup.id] = cup

  def process_detections(self, detections):
    for det in detections:
      pos = det[:4]

      # Detected ball 
      if (det[5] == BALL_CLASS):
        
        # Start position
        if (self.ball_start is None):
          self.ball_start = pos

        # Detected final ball position
        if ((self.ball_start is not None) and (self.ball_count > 60)):
          self.ball_end = pos
          # Decide under which cup the ball is right now (at the end of the game)
          for cup_id in self.cups.keys():
            cup = self.cups[cup_id]
            
            if not self.prediction_made:
              # Predict ball position
              if ((cup.positions[-1][0] < self.ball_end[0]) and (cup.positions[-1][2] > self.ball_end[2])):
                self.cup_with_ball_true = cup.id
                self.prediction_made = True

              # Decide if prediction was correct
              if (self.cup_with_ball_true == self.cup_with_ball):
                self.message = 'final position of ball detected under cup {}\nour prediction was {}\nCORRECT'.format(
                  self.cup_with_ball_true, self.cup_with_ball)
                self.message_count = 0
                self.correct = 1
              else:
                self.message = 'final position of ball detected under cup {}\nour prediction was {}\nWRONG'.format(
                  self.cup_with_ball_true, self.cup_with_ball)
                self.message_count = 0


      # Ball position already detected, start detecting cups
      # Cup detected
      if (self.ball_start is not None) and (det[5] == CUP_CLASS):
        cup_id = det[4]
        self.ball_count

        # Not yet decided, under which cup the ball was at the start
        if (self.cup_with_ball == -1):
          if ((pos[0] < self.ball_start[0]) and (pos[2] > self.ball_start[2])):
            self.cup_with_ball = cup_id
            self.message = 'initial position of ball detected under cup {}'.format(cup_id)
            self.message_count = 0

        if cup_id not in self.cups.keys(): # New cup detected - potentially some cup that was previously lost
          cup = Cup(cup_id, pos)
          self.add_cup(cup)
        else:
          self.cups[cup_id].update(pos)

    # Increment counters and return message to be displayed in video.
    self.message_count += 1
    self.ball_count += 1
    if self.message_count > 30:
      self.message = None
    

    return self.message, self.correct


def detect(opt):
  out, source, yolo_model, deep_sort_model, show_vid, save_vid, save_txt, imgsz, evaluate, half, project, name, exist_ok= \
    opt.output, opt.source, opt.yolo_model, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
    opt.save_txt, opt.imgsz, opt.evaluate, opt.half, opt.project, opt.name, opt.exist_ok
  webcam = source == '0' or source.startswith(
    'rtsp') or source.startswith('http') or source.endswith('.txt')

  # Init CupDetector
  cup_detector = CupDetector()
  cup_detector_message = None

  # initialize deepsort
  cfg = get_config()
  cfg.merge_from_file(opt.config_deepsort)
  deepsort = DeepSort(deep_sort_model,
            max_dist=cfg.DEEPSORT.MAX_DIST,
            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
            use_cuda=True)

  # Initialize
  device = select_device(opt.device)
  half &= device.type != 'cpu'  # half precision only supported on CUDA

  # The MOT16 evaluation runs multiple inference streams in parallel, each one writing to
  # its own .txt file. Hence, in that case, the output folder is not restored
  if not evaluate:
    if os.path.exists(out):
      pass
      shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

  # Directories
  save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
  save_dir.mkdir(parents=True, exist_ok=True)  # make dir

  # Load model
  device = select_device(device)
  model = DetectMultiBackend(yolo_model, device=device, dnn=opt.dnn)
  stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
  imgsz = check_img_size(imgsz, s=stride)  # check image size

  # Half
  half &= pt and device.type != 'cpu'  # half precision only supported by PyTorch on CUDA
  if pt:
    model.model.half() if half else model.model.float()

  # Set Dataloader
  vid_path, vid_writer = None, None
  

  # Check if environment supports image displays
  if show_vid:
    show_vid = check_imshow()

  # Dataloader
  if webcam:
    show_vid = check_imshow()
    cudnn.benchmark = True  # set True to speed up constant image size inference
    dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    bs = len(dataset)  # batch_size
  else:
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt and not jit)
    bs = 1  # batch_size
  vid_path, vid_writer = [None] * bs, [None] * bs

  # Get names and colors
  names = model.module.names if hasattr(model, 'module') else model.names

  # extract what is in between the last '/' and last '.'
  txt_file_name = source.split('/')[-1].split('.')[0]
  txt_path = str(Path(save_dir)) + '/' + txt_file_name + '.txt'

  if pt and device.type != 'cpu':
    model(torch.zeros(1, 3, *imgsz).to(device).type_as(next(model.model.parameters())))  # warmup
  dt, seen = [0.0, 0.0, 0.0, 0.0], 0
  for frame_idx, (path, img, im0s, vid_cap, s) in enumerate(dataset):

    if opt.max_frame > 0:
      if frame_idx > opt.max_frame:
        break

    t1 = time_sync()
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
      img = img.unsqueeze(0)
    t2 = time_sync()
    dt[0] += t2 - t1

    # Inference
    visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if opt.visualize else False
    pred = model(img, augment=opt.augment, visualize=visualize)
    t3 = time_sync()
    dt[1] += t3 - t2

    # Apply NMS
    if '.mlmodel' not in opt.yolo_model[0]:
      pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
    dt[2] += time_sync() - t3

    if '.mlmodel' in opt.yolo_model[0]:
      pred = [pred]
    # Process detections
    for i, det in enumerate(pred):  # detections per image
      seen += 1
      if webcam:  # batch_size >= 1
        p, im0, _ = path[i], im0s[i].copy(), dataset.count
        s += f'{i}: '
      else:
        p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)

      p = Path(p)  # to Path
      save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
      s += '%gx%g ' % img.shape[2:]  # print string

      annotator = Annotator(im0, line_width=2, pil=not ascii)

      if det is not None and len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(
          img.shape[2:], det[:, :4], im0.shape).round()

        # Print results
        for c in det[:, -1].unique():
          n = (det[:, -1] == c).sum()  # detections per class
          s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

        xywhs = xyxy2xywh(det[:, 0:4])
        confs = det[:, 4]
        clss = det[:, 5]

        # pass detections to deepsort
        t4 = time_sync()
        outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
        m, correct_prediction = cup_detector.process_detections(outputs)
        if m is not None:
          cup_detector_message = m

        t5 = time_sync()
        dt[3] += t5 - t4

        # draw boxes for visualization
        if len(outputs) > 0:
          for j, (output, conf) in enumerate(zip(outputs, confs)):

            bboxes = output[0:4]
            id = output[4]
            cls = output[5]

            c = int(cls)  # integer class
            label = f'{id} {names[c]} {conf:.2f}'
            annotator.box_label(bboxes, label, color=colors(c, True))

            if save_txt:
              # to MOT format
              bbox_left = output[0]
              bbox_top = output[1]
              bbox_w = output[2] - output[0]
              bbox_h = output[3] - output[1]
              # Write MOT compliant results to file
              with open(txt_path, 'a') as f:
                f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                 bbox_top, bbox_w, bbox_h, -1, -1, -1, -1))

        LOGGER.info(f'{s}Done. YOLO:({t3 - t2:.3f}s), DeepSort:({t5 - t4:.3f}s)')

      else:
        deepsort.increment_ages()
        LOGGER.info('No detections')

      # Stream results
      im0 = annotator.result()
      if show_vid:
        if cup_detector_message is not None:
          # cv2.putText(im0, cup_detector_message, (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
          
          y0, dy = 50, 50
          
          # Print message
          for i, line in enumerate(cup_detector_message.split('\n')):
            y = y0 + i*dy
            cv2.putText(im0, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        cv2.imshow(str(p), im0)
        if cv2.waitKey(1) == ord('q'):  # q to quit
          raise StopIteration

      # Save results (image with detections)
      if save_vid:
        if vid_path != save_path:  # new video
          vid_path = save_path
          if isinstance(vid_writer, cv2.VideoWriter):
            vid_writer.release()  # release previous video writer
          if vid_cap:  # video
            fps = vid_cap.get(cv2.CAP_PROP_FPS)
            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
          else:  # stream
            fps, w, h = 30, im0.shape[1], im0.shape[0]

          vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        if cup_detector_message is not None:
          y0, dy = 50, 50
            
          # Print message
          for i, line in enumerate(cup_detector_message.split('\n')):
            y = y0 + i*dy
            cv2.putText(im0, line, (50, y ), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

        vid_writer.write(im0)

  # Print results
  t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
  LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms deep sort update \
    per image at shape {(1, 3, *imgsz)}' % t)
  if save_txt or save_vid:
    print('Results saved to %s' % save_path)
    if platform == 'darwin':  # MacOS
      os.system('open ' + save_path)

  return cup_detector

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--yolo_model', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
  parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
  parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
  parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
  parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
  parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
  parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
  parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
  parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
  parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
  parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
  parser.add_argument('--save-txt', action='store_true', help='save MOT compliant results to *.txt')
  # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
  parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
  parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
  parser.add_argument('--augment', action='store_true', help='augmented inference')
  parser.add_argument('--evaluate', action='store_true', help='augmented inference')
  parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
  parser.add_argument("--half", action="store_true", help="use FP16 half-precision inference")
  parser.add_argument('--visualize', action='store_true', help='visualize features')
  parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
  parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
  parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
  parser.add_argument('--name', default='exp', help='save results to project/name')
  parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
  parser.add_argument('--max_frame', type=int, default=0)
  opt = parser.parse_args()
  opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand

  opt.classes = [BALL_CLASS, CUP_CLASS]


  if '.mlmodel' in opt.yolo_model[0]:
    opt.imgsz = [320, 192]

  cup_detector = detect(opt)

