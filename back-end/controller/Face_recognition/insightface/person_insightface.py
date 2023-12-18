import sys
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import onnxruntime as ort
import cv2

class PersonAnalyzer:
    def __init__(self):
        self.detectorPerson = insightface.model_zoo.get_model(r"D:\PROJEC_THANGLT\insightface\model_zoo\buffalo_l\scrfd_person_2.5g.onnx", download=False)
        self.detectorPerson.prepare(0, nms_thresh=0.5, input_size=(640, 640))

    # def age_gender_analysis(self, img):
    #     list_age_gender = []
    #     faces = self.age_gender_model.get(img)
    #     # assert len(faces)==6
    #     for face in faces:
    #         list_age_gender.append([face.bbox, face.age, face.sex])
    #     return list_age_gender
    def detect_person(self,img):
        bboxes, kpss = self.detectorPerson.detect(img)
        bboxes = np.round(bboxes[:,:4]).astype(np.int64)
        kpss = np.round(kpss).astype(np.int64)
        kpss[:,:,0] = np.clip(kpss[:,:,0], 0, img.shape[1])
        kpss[:,:,1] = np.clip(kpss[:,:,1], 0, img.shape[0])
        vbboxes = bboxes.copy()
        vbboxes[:,0] = kpss[:, 0, 0]
        vbboxes[:,1] = kpss[:, 0, 1]
        vbboxes[:,2] = kpss[:, 4, 0]
        vbboxes[:,3] = kpss[:, 4, 1]
        return bboxes, vbboxes
    def draw_person(self, img, bboxes, vbboxes):
        bboxes, vbboxes = self.detect_person(img, self.detectorPerson)
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            vbbox = vbboxes[i]
            x1,y1,x2,y2 = bbox
            vx1,vy1,vx2,vy2 = vbbox
            cv2.rectangle(img, (x1,y1)  , (x2,y2) , (0,255,0) , 1)
            alpha = 0.8
            color = (255, 0, 0)
            for c in range(3):
                img[vy1:vy2,vx1:vx2,c] = img[vy1:vy2, vx1:vx2, c]*alpha + color[c]*(1.0-alpha)
            cv2.circle(img, (vx1,vy1) , 1, color , 2)
            cv2.circle(img, (vx1,vy2) , 1, color , 2)
            cv2.circle(img, (vx2,vy1) , 1, color , 2)
            cv2.circle(img, (vx2,vy2) , 1, color , 2)
        return img