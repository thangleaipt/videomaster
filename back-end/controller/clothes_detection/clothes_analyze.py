import torch
import os
import cv2
from controller.clothes_detection.yolo.utils.utils import *
from controller.clothes_detection.predictors.YOLOv3 import YOLOv3Predictor
import glob
from tqdm import tqdm
import sys
import uuid
from config import WEIGHT_FOLDER
import pandas as pd

class CLOTHESANALYZE():
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.cuda.empty_cache()
        self.yolo_modanet_params = {   "model_def" : os.path.join(WEIGHT_FOLDER,"clothes_model/modanetcfg/yolov3-modanet.cfg"),
            "weights_path" : os.path.join(WEIGHT_FOLDER, "clothes_model/weights/yolov3-modanet_last.weights"),
            "class_path": os.path.join(WEIGHT_FOLDER, "clothes_model/modanetcfg/modanet.names"),
            "conf_thres" : 0.5,
            "nms_thres" :0.5,
            "img_size" : 416,
            "device" : self.device}
        self. detectron = YOLOv3Predictor(params=self.yolo_modanet_params)
        self.classes = load_classes(self.yolo_modanet_params["class_path"])
        index=["color","color_name","hex","R","G","B"]
        self.ref_color = pd.read_csv(os.path.join(WEIGHT_FOLDER, "colors_vn.csv"), names=index, header=None)

    def find_main_rgb_from_roi(self,img_roi):
        """
        This function will find out the major color component for a given image ROI
        :param img_roi: sub image
        :return: main color component in RGB tuple (R, G, B)
        """
        if img_roi.shape[0] > 32 or img_roi.shape[1] > 32:
            img_roi = cv2.resize(img_roi, (32, 32))
        Z = img_roi.reshape((-1, 3))

        # convert to np.float32
        Z = np.float32(Z)
        # Perform K-means clustering using NumPy
        K = 3
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # Find the cluster with the maximum number of pixels
        pixel_counts = np.bincount(label.flatten())
        index_max_hist = np.argmax(pixel_counts)

        # Get the main RGB component
        rgb = center[index_max_hist]
        main_rgb = tuple((int(rgb[0]), int(rgb[1]), int(rgb[2])))
        return main_rgb
    "00:'bag'01:'belt'02:'boots'03:'footwear'04:'outer'05:'dress'06:'sunglasses'07:'pants'08:'top'09:'shorts'10:'skirt'11:'headwear'12'neckwear'"
    def clothes_detector(self,img):
        try:
            is_check_glasses = None
            name = None
            max_conf = 0
            detections = self.detectron.get_detections(img)
            main_color = None
            if len(detections) != 0 :
                detections.sort(reverse=False ,key = lambda x:x[4])
                for x1, y1, x2, y2, cls_conf, cls_pred in detections:
                    if cls_pred in [4,5,7,8,9,10]:
                        if cls_conf > max_conf:
                            max_conf = cls_conf
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            # making a sub image
                            width_difference = x2 - x1
                            height_difference = y2 - y1
                            img_sub = img[max(0,y1 + int(height_difference / 7)):min(img.shape[0],y2 - int(height_difference / 7)),
                                    max(0,x1 + int(width_difference / 7)):min(img.shape[1],x2 - int(width_difference / 7))]
                            if img_sub is not None and len(img_sub) > 0:
                                main_color = self.find_main_rgb_from_roi(img_sub)
                            else:
                                main_color = None
            # if main_color is not None:
            #     name = self.min_color_diff(main_color)       
            return main_color,name
        except Exception as e:
            print("[clothes_analyze][clothes_detector]: ", e)
            return None, False
        
    # def color_dist( self,c1, c2):
    #         """ returns the squared euklidian distance between two color vectors in yuv space """
    #         return sum((a-b)**2 for a,b in zip(self.to_ycc(c1),self.to_ycc(c2)))
    def bgr_to_yuv(self,bgr_color):
        """Converts a BGR color to YUV color space."""
        b, g, r = bgr_color
        
        y = 0.299 * r + 0.587 * g + 0.114 * b
        u = -0.14713 * r - 0.288862 * g + 0.436 * b
        v = 0.615 * r - 0.51498 * g - 0.10001 * b
        
        return y, u, v
    def color_dist(self, c1, c2):
        """Returns the squared Euclidean distance between two color vectors in YUV space."""
        
        # Convert BGR to YUV for both input colors
        yuv_c1 = self.bgr_to_yuv(c1)
        yuv_c2 = self.bgr_to_yuv(c2)
        
        # Calculate the squared Euclidean distance
        return sum((a - b) ** 2 for a, b in zip(yuv_c1, yuv_c2))

    def min_color_diff(self, color_to_match):
        color_array = self.ref_color[["B", "G", "R"]].values
        color_to_match = np.array(color_to_match)
        
        # Chuyển đổi màu cần so sánh sang không gian YUV
        yuv_color_to_match = self.bgr_to_yuv(color_to_match)
        
        # Chuyển đổi toàn bộ màu trong self.ref_color sang không gian YUV
        yuv_colors = np.apply_along_axis(self.bgr_to_yuv, axis=1, arr=color_array)
        
        # Tính khoảng cách cho tất cả các màu cùng một lúc
        differences = yuv_colors - yuv_color_to_match
        distances = np.linalg.norm(differences, axis=1)
        
        min_distance_idx = np.argmin(distances)
        
        return self.ref_color.loc[min_distance_idx, "color_name"]
    