from pathlib import Path
import time
import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from controller.boxmot import create_tracker
from config import WEIGHT_FOLDER

class Tracker:
    def __init__(self,detector_weights, device):

        self.device = torch.device(device)

        self.model_detector = YOLO(detector_weights)
        self.model_detector.to(self.device)

        self.tracker = create_tracker(
            tracker_type='strongsort',
            reid_weights=Path(os.path.join(WEIGHT_FOLDER, 'osnet_x0_25_msmt17.pt')),
            tracker_config=Path(os.path.join(WEIGHT_FOLDER, 'configs/strongsort.yaml')),
            device=self.device,
            half=False,
            per_class=True
        )

    def tracking_frame(self,frame):
        results = self.model_detector(
            frame,
            classes=[0],
            conf=0.3,
            verbose=False
        )
        pred = results[0].boxes
        pred_np = pred.data.cpu().numpy()

        track_result = self.tracker.update(pred_np, frame)
        result = []
        for r in track_result:
            xyxy = r[0:4].astype(np.int32)
            id = int(r[4])
            result.append([xyxy[0],xyxy[1],xyxy[2],xyxy[3], id])

        return result
