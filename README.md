# FaceMask_WobotAI
**Please visit master branch
FaceMask Classification using YOLOv8x-cls

train : docker run --rm -v D:\FaceMask_WobottAI yolo_app train --data D:\FaceMask_WobottAI\dataset\data --epochs 10 --imgsz 640
inference : docker run --rm -v D:\FaceMask_WobottAI yolo_app infer --model model/best.pt --video FaceMask/Test_video2.mp4 --save
