
import sys
from ultralytics import YOLO
m = YOLO("yolov8n.yaml")
m.train(data=sys.argv[1], imgsz=640, epochs=100000, batch=16, workers=8, seed=0, device=0, amp=True, val=False, plots=False,
        patience=100000, save=False, project=sys.argv[2], name="tenant", exist_ok=True, verbose=False)
