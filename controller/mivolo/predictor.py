import os

from controller.tracking_analyze import Tracker

from controller.clothes_detection.clothes_analyze import CLOTHESANALYZE

from controller.mask_detection.mask_analyze import MaskDetector
from config import DEVICE, WEIGHT_FOLDER

import cv2
import numpy as np

from controller.mivolo.structures import PersonAndFaceResult

class Predictor:
    def __init__(self):
        # Trigger enable person_face_detector or vehicle_detector
        self.is_face_body_detect_model = True

        device = DEVICE
        self.list_distance_color = []

        detector_weights = os.path.join(WEIGHT_FOLDER, "yolov8s.pt")
        self.person_tracking = Tracker(detector_weights, device)
        self.draw = True
        self.clothes_model = CLOTHESANALYZE()
    
    def analyze_tracking_frame(self, image: np.ndarray) -> PersonAndFaceResult:
        try:
            frame = image.copy()
            detected_objects: PersonAndFaceResult = self.person_face_detector.track(frame)
            list_box = []
            pred_boxes = detected_objects.yolo_results.boxes
            for _, (det) in enumerate(pred_boxes):
                box = det.xyxy.squeeze().tolist()

                # Clothes detection
                person_image = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                main_color_clothes, name_color = self.clothes_model.clothes_detector(person_image)
                list_box.append([int(i) for i in box])

            yield detected_objects,list_box

        except Exception as e:
            print(f"[analyze_tracking_frame][recognition]: {e}")
            yield None, None

    def analyze_tracking_boxmot(self, image: np.ndarray):
        try:
            frame = image.copy()
            list_track_results = []
            track_result = self.person_tracking.tracking_frame(frame)
            for track in track_result:
                box = track[0:4]
                # Clothes detection
                person_image = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                main_color_clothes, name_color = self.clothes_model.clothes_detector(person_image)
                list_track_results.append([track, main_color_clothes, name_color])
            return list_track_results
        except Exception as e:
            print(f"[analyze_tracking_boxmot][recognition]: {e}")
            return []


if __name__ == "__main__":
  predictor = Predictor()
  path_image = r"C:\Users\Admin\Pictures\Camera Roll\photo_2023-10-02_15-26-30.jpg"
  frame = cv2.imread(path_image)
  detected_objects_history, frame, detected_objects = predictor.analyze_tracking_frame(predictor, frame)
  print(detected_objects_history)
