import os
from typing import Dict, Union
import cv2

import numpy as np
import PIL
import torch
from controller.mivolo.structures import PersonAndFaceResult
from ultralytics import YOLO
from collections import defaultdict
from config import WEIGHT_FOLDER

class Detector:
    def __init__(
        self,
        weights: str,
        device: str = "cuda",
        half: bool = True,
        verbose: bool = False,
        conf_thresh: float = 0.5,
        iou_thresh: float = 0.5,
        is_face_body_detect_model: bool = False
    ):
        self.is_face_body_detect_model = is_face_body_detect_model
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.verbose = verbose

        self.vehicle_track_history = defaultdict(lambda: [])
        self.list_count_vehicles = []
        self.list_vehicles = []
        self.list_check_id_vehicles = []
        self.yolo = YOLO(weights)
        
        self.yolo.fuse()

        self.device = torch.device(device)
        self.half = half and self.device.type != "cpu"

        if self.half:
            self.yolo.model = self.yolo.model.half()

        self.detector_names: Dict[int, str] = self.yolo.model.names

        # init yolo.predictor
        # classes=[0,1]
        self.classes = 0
        self.detector_kwargs = {"conf": conf_thresh, "iou": iou_thresh, "half": self.half, "verbose": verbose, 'device': self.device, "classes": self.classes}
        # self.yolo.predict(**self.detector_kwargs)

    def predict(self, image: Union[np.ndarray, str, "PIL.Image"]) -> PersonAndFaceResult:
        results = self.yolo.predict(image, **self.detector_kwargs)[0]
        return PersonAndFaceResult(results)

    def track(self, image: Union[np.ndarray, str, "PIL.Image"]) -> PersonAndFaceResult:
        # if self.is_face_body_detect_model:
        #     self.detector_kwargs = {"conf": self.conf_thresh, "iou": self.iou_thresh, "half": self.half, "verbose": self.verbose,"classes": 0 }
        results = self.yolo.track(image, persist=True, **self.detector_kwargs)[0]
        return PersonAndFaceResult(results)
    
    def vehicle_track(self, image: Union[np.ndarray, str, "PIL.Image"]):
        results = self.yolo.track(image, persist=True)
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        cls = results[0].boxes.cls.int().cpu().tolist()
        list_vehicles = []
        # Visualize the results on the frame
        

        #line: (400, 660) (1580,660)
        start_threshold = (400, 500)
        end_threshold = (1580, 500)
        # Plot the tracks
        for box, track_id, cls in zip(boxes, track_ids, cls):
            x, y, w, h = box
            track = self.vehicle_track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 50:  # retain 90 tracks for 90 frames
                track.pop(0)

            list_vehicles.append([box,track_id, cls])
            if track_id not in self.list_check_id_vehicles:
                self.list_check_id_vehicles.append(track_id)
            if float(y) > start_threshold[1] and track_id not in self.list_count_vehicles:
                self.list_count_vehicles.append(track_id)
                

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.putText(image, f'Counting: {len(self.list_count_vehicles)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 2)
            #cv2.polylines(annotated_frame, [points], isClosed=False, color=(255, 255, 0), thickness=10)
            # annotated_frame = results[0].plot()
            cv2.line(image, start_threshold, end_threshold, (0, 0, 255), 2)
        return image, list_vehicles
