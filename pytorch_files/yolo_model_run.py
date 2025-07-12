import os
import sys
import argparse
import glob
import time
from collections import deque
from picamera2 import Picamera2

import cv2
import numpy as np
from ultralytics import YOLO

picam = None

parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True)
parser.add_argument('--source', required=True)
parser.add_argument('--thresh', default=0.5)
parser.add_argument('--resolution', default=None)
parser.add_argument('--record', action='store_true')
args = parser.parse_args()

model_path = args.model
img_source = args.source
min_thresh = float(args.thresh)
user_res = args.resolution
record = args.record

if not os.path.exists(model_path):
    print('ERROR: Model path is invalid.')
    sys.exit(1)

model = YOLO(model_path)
labels = model.names

img_ext = ['.jpg','.jpeg','.png','.bmp']
vid_ext = ['.avi','.mov','.mp4','.mkv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    source_type = 'image' if ext in img_ext else 'video' if ext in vid_ext else sys.exit('Unsupported file extension')
elif img_source.startswith('usb'):
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif img_source == 'picamera':
    source_type = 'picamera'
else:
    print('Invalid source input.')
    sys.exit(1)

resize = False
if user_res:
    resize = True
    resW, resH = map(int, user_res.split('x'))

record_name = 'demo1.avi'
record_fps = 30
recorder = None

if source_type == 'picamera':
    picam = Picamera2()
    picam.configure(picam.create_video_configuration(main={'format': 'RGB888', 'size': (resW, resH)}))
    picam.start()
elif source_type in ['video', 'usb']:
    cap_arg = img_source if source_type == 'video' else usb_idx
    cap = cv2.VideoCapture(cap_arg)
    if user_res:
        cap.set(3, resW)
        cap.set(4, resH)
elif source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = [f for f in glob.glob(img_source + '/*') if os.path.splitext(f)[1].lower() in img_ext]

bbox_colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0)]

img_count = 0
fps_history = deque(maxlen=30)

while True:
    t_start = time.perf_counter()

    if source_type in ['image', 'folder']:
        if img_count >= len(imgs_list): break
        frame = cv2.imread(imgs_list[img_count])
        img_count += 1
    elif source_type in ['video', 'usb']:
        ret, frame = cap.read()
        if not ret: break
    elif source_type == 'picamera':
        frame = picam.capture_array()

    if resize:
        frame = cv2.resize(frame, (resW, resH))

    results = model(frame, verbose=False)
    detections = results[0].boxes

    for det in detections:
        xyxy = det.xyxy.cpu().numpy().astype(int).squeeze()
        conf = det.conf.item()
        cls = int(det.cls.item())
        if conf > min_thresh:
            cv2.rectangle(frame, tuple(xyxy[:2]), tuple(xyxy[2:]), bbox_colors[cls % len(bbox_colors)], 2)
            label = f"{labels[cls]}: {conf:.2f}"
            cv2.putText(frame, label, (xyxy[0], xyxy[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    t_end = time.perf_counter()
    fps = 1.0 / (t_end - t_start)
    fps_history.append(fps)
    avg_fps = sum(fps_history) / len(fps_history)

    cv2.putText(frame, f"FPS: {fps:.2f} (avg: {avg_fps:.2f})", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if record and source_type in ['video', 'usb', 'picamera']:
        if recorder is None:
            recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))
        recorder.write(frame)

    cv2.imshow('YOLO Detection', frame)
    key = cv2.waitKey(5) & 0xFF
    if key == ord('q'): break

if source_type in ['video', 'usb']: cap.release()
elif source_type == 'picamera': picam.stop()
if recorder: recorder.release()
cv2.destroyAllWindows()
