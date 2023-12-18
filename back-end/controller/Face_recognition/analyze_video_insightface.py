from collections import Counter
from datetime import datetime
import os
import queue
import threading
import winsound
import pygame
from unidecode import unidecode
import tkinter as tk
from PIL import Image, ImageTk

from gtts import gTTS

from controller.mivolo.person_model import PersonModel

from server.config import DATE_TIME_FORMAT, DATETIME_FORMAT

from controller.Face_recognition.report import send_report_to_db

from controller.mivolo.data.misc import prepare_classification_images
from controller.mivolo.model.mi_volo import MiVOLO

import cv2
import numpy as np
import onnxruntime
from controller.Face_recognition.insightface.scrfd import SCRFD
from controller.Face_recognition.insightface.arcface_onnx import ArcFaceONNX
from os import path
import pickle
from tqdm import tqdm
import pandas as pd
import time
from config import DEVICE, STATIC_FOLDER, WEIGHT_FOLDER, DATASET_PEOPLE_FOLDER
from unidecode import unidecode

from controller.mask_detection.mask_analyze import MaskDetector
from controller.mivolo.predictor import Predictor
from PySide2.QtCore import QThread

onnxruntime.set_default_logger_severity(3)
class FaceAnalysisInsightFace:
    def __init__(self, path = ""):
        self.video_path = path
        self.assets_dir = os.path.join(WEIGHT_FOLDER,"antelopev2")
        self.db_path = DATASET_PEOPLE_FOLDER
        device = DEVICE
        self.detector = SCRFD(os.path.join(self.assets_dir, 'det_10g.onnx'))
        self.detector.prepare(0)
        self.model_path = os.path.join(self.assets_dir, 'glintr100.onnx')
        self.rec = ArcFaceONNX(self.model_path)
        self.rec.prepare(0)
        self.encoding_features = []
        self.list_labels = []
        self.list_person_trained = []
        self.list_labels_position = []
        self.list_label_disable = {}
        self.labels_to_remove = []

        self.representations = self.load_db_from_folder()
        mask_model_path = os.path.join(WEIGHT_FOLDER, "mask_model/Res18oneFC_model.pth")
        self.mask_analyze = MaskDetector(mask_model_path)
        self.person_analyze = Predictor()
        checkpoint_age_gender_model = os.path.join(WEIGHT_FOLDER, "model_imdb_age_gender_4.22.pth.tar")
        self.age_gender_model = MiVOLO(checkpoint_age_gender_model,device,half=True,use_persons=False,disable_faces=False,verbose=False)
        self.index_frame = 0
        self.list_label_greeting = []
        
        self.buffer_label = queue.Queue()
        self.list_image_label = []
        self.is_running_greeting = True
        self.feature_image = []
        
    def load_db_from_folder(self):
        try:
            db_path = self.db_path

            print(f"Loading representations from {db_path}")

            file_name = f"representations_{os.path.basename(self.model_path)}.pkl"
            file_name = file_name.replace("-", "_").lower()

            if path.exists(db_path + "/" + file_name):
                list_trained_path = []
                with open(f"{db_path}/{file_name}", "rb") as f:
                    representations = pickle.load(f)
                    for rep in representations:
                        list_trained_path.append(os.path.normpath(rep[0]))
                        name = os.path.basename(rep[0])
                        label = os.path.splitext(name)[0]
                        self.list_labels.append(rep[2])
                        self.list_labels_position.append(rep[3])
                        self.list_label_disable = dict.fromkeys(self.list_labels, 0)
                        self.encoding_features.append(rep[1])
                
                #     for item in os.listdir(db_path):
                #         item_path = os.path.join(db_path, item)
                #         if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                #             if os.path.normpath(item_path) not in list_trained_path:
                #                 list_train_path.append(item_path)

                #         elif os.path.isdir(item_path):
                #             for _, _, sub_files in os.walk(item_path):
                #                 for sub_file in sub_files:
                #                     if (
                #                         sub_file.lower().endswith((".jpg", ".jpeg", ".png"))
                #                     ):
                #                         sub_file_path = os.path.join(item_path, sub_file)
                #                         if sub_file_path not in list_trained_path:
                #                             list_sub_train_path.append(sub_file_path)

                #     pbar = tqdm(
                #     range(0, len(list_train_path)),
                #     desc="Finding representations",
                #     disable=True,
                # )
                #     sub_pbar = tqdm(
                #     range(0, len(list_sub_train_path)),
                #     desc="Finding representations",
                #     disable=True,
                # )
                    
                #     for index in pbar:
                #         employee = list_train_path[index]
                #         frame = cv2.imread(employee)
                #         bboxes1, kpss1 = self.detector.autodetect(frame, max_num=1)
                #         kps1 = kpss1[0]
                #         img_representation = self.rec.get(frame, kps1)

                #         instance = []
                #         instance.append(employee)
                #         instance.append(img_representation)

                #         path_folder = os.path.dirname(employee)
                #         label = os.path.basename(path_folder)
                #         instance.append(label)
                #         representations.append(instance)
                #         self.list_labels.append(label)
                #         self.encoding_features.append(img_representation)

                #     for index in sub_pbar:
                #         employee = list_sub_train_path[index]
                #         frame = cv2.imread(employee)
                #         bboxes1, kpss1 = self.detector.autodetect(frame, max_num=1)
                #         kps1 = kpss1[0]
                #         img_representation = self.rec.get(frame, kps1)

                #         instance = []
                #         instance.append(employee)
                #         instance.append(img_representation)

                #         path_folder = os.path.dirname(employee)
                #         label = os.path.basename(path_folder)
                #         instance.append(label)
                #         representations.append(instance)
                #         self.list_labels.append(label)
                #         self.encoding_features.append(img_representation)

                with open(f"{db_path}/{file_name}", "wb") as f:
                    pickle.dump(representations, f)          


            else:
                employees = []
                sub_employees = []

                # for item in os.listdir(db_path):
                #     item_path = os.path.join(db_path, item)
                #     if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                #         mask_image = add_mask(item_path)
                
                #     elif os.path.isdir(item_path):
                #         for _, _, sub_files in os.walk(item_path):
                #             for sub_file in sub_files:
                #                 if (
                #                     sub_file.lower().endswith((".jpg", ".jpeg", ".png"))
                #                 ):
                #                     sub_file_path = os.path.join(item_path, sub_file)
                #                     mask_image = add_mask(sub_file_path) 

                # for item in os.listdir(db_path):
                #     item_path = os.path.join(db_path, item)
                #     if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                #         employees.append(item_path)

                #     elif os.path.isdir(item_path):
                #         for _, _, sub_files in os.walk(item_path):
                #             for sub_file in sub_files:
                #                 if (
                #                     sub_file.lower().endswith((".jpg", ".jpeg", ".png"))
                #                 ):
                #                     sub_file_path = os.path.join(item_path, sub_file)
                #                     sub_employees.append(sub_file_path)

                # find representations for db images
                representations = []

                # for employee in employees:
                # pbar = tqdm(
                #     range(0, len(employees)),
                #     desc="Finding representations",
                #     disable=True,
                # )

                # sub_pbar = tqdm(
                #     range(0, len(sub_employees)),
                #     desc="Finding representations",
                #     disable=True,
                # )
                # for index in range(0, len(employees)):
                #     employee = employees[index]
                #     frame = cv2.imread(employee)
                #     bboxes1, kpss1 = self.detector.autodetect(frame, max_num=1)
                #     kps1 = kpss1[0]
                #     img_representation = self.rec.get(frame, kps1)

                #     instance = []
                #     instance.append(employee)
                #     instance.append(img_representation)

                #     path_folder = os.path.dirname(employee)
                #     label = os.path.basename(path_folder)
                #     instance.append(label)
                #     representations.append(instance)
                #     self.list_labels.append(label)
                #     self.encoding_features.append(img_representation)

                # for index in sub_pbar:
                #     employee = sub_employees[index]
                #     frame = cv2.imread(employee)
                #     bboxes1, kpss1 = self.detector.autodetect(frame, max_num=1)
                #     kps1 = kpss1[0]
                #     img_representation = self.rec.get(frame, kps1)

                #     instance = []
                #     instance.append(employee)
                #     instance.append(img_representation)

                #     path_folder = os.path.dirname(employee)
                #     label = os.path.basename(path_folder)
                #     instance.append(label)
                #     representations.append(instance)
                #     self.list_labels.append(label)
                #     self.encoding_features.append(img_representation)
                    

                # -------------------------------
                with open(f"{db_path}/{file_name}", "wb") as f:
                    pickle.dump(representations, f)
            self.representations = representations
            return representations
        except Exception as e:
            print(f"[analyze_video_insightface][train]: {e}")
            return []
    
    def train(self, image, label, label_position):
        try:
            db_path = self.db_path
            file_name = f"representations_{os.path.basename(self.model_path)}.pkl"
            file_name = file_name.replace("-", "_").lower()
            if path.exists(db_path + "/" + file_name):
                print(
                    f"WARNING: Representations for images in {db_path} folder were previously stored"
                    + f" in {file_name}. If you added new instances after the creation, then please "
                    + "delete this file and call find function again. It will create it again."
                )

                with open(f"{db_path}/{file_name}", "rb") as f:
                    representations = pickle.load(f)
            
                bboxes1, kpss1 = self.detector.autodetect(image, max_num=1)
                kps1 = kpss1[0]
                img_representation = self.rec.get(image, kps1)

                # check path image
                image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
                image_file_found = False
                image_file_path = None
                label_imcode = unidecode(label).replace(" ", "_").lower()
                dir_save_image = f"{db_path}/{label_imcode}"
                if not os.path.exists(dir_save_image):
                    os.makedirs(dir_save_image)
                path_save_image = f"{dir_save_image}/{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                for extension in image_extensions:
                    image_file_path = path_save_image.replace(".jpg", f".{extension}")
                    if os.path.exists(image_file_path):
                        image_file_found = True
                        break
                if image_file_found and image_file_path is not None:
                    # delete image
                    os.remove(image_file_path)
                
                cv2.imwrite(path_save_image, image)

                representations.append([path_save_image, img_representation, label, label_position])
                self.representations = representations
                self.list_labels.append(label)
                self.list_labels_position.append(label_position)
                self.encoding_features.append(img_representation)

                with open(f"{db_path}/{file_name}", "wb") as f:
                    f.seek(0)
                    f.truncate()
                    pickle.dump(representations, f)
            
            else:
                self.load_db_from_folder()
                
        except Exception as e:
            representations = []
    
    def delete(self, label):
        try:
            file_name = f"representations_{os.path.basename(self.model_path)}.pkl"
            file_name = file_name.replace("-", "_").lower()

            if label not in self.list_labels:
                print(f'{label} does not exist in dataset')
                return
            indices = [i for i, x in enumerate(self.list_labels) if x == label]

            if len(indices) == 0:
                print(f'{label} does not exist in dataset')
                return
            

            # for idx in indices:
            if type(self.representations) == list:
                for idx in sorted(indices, reverse=True):
                    del self.representations[idx]
            else:
                self.representations = np.delete(self.representations, indices, axis=0)
            if type(self.list_labels) == list:
                for idx in sorted(indices, reverse=True):
                    del self.list_labels[idx]
            else:
                self.list_labels = np.delete(self.list_labels, indices)

            if type(self.encoding_features) == list:
                for idx in sorted(indices, reverse=True):
                    del self.encoding_features[idx]
            else:
                self.encoding_features = np.delete(self.encoding_features, indices, axis=0)
            print(f'encodings: {len(self.representations)}')
            print(f'labels: {len(self.list_labels)}')
            with open(f"{self.db_path}/{file_name}", "wb") as f:
                f.seek(0)
                f.truncate()
                pickle.dump(self.representations, f)
            print(f"delete {label} from dataset")
        except Exception as e:
            print('delete label fail', e)
           

    def recognition(self, target_img, box, kpss1):
        try:
            representations = self.representations
            df = pd.DataFrame(representations, columns=["identity", f"{os.path.basename(self.model_path)}_representation","labels","position"])
            result_df = df.copy()  # df will be filtered in each img
            target_representation = self.rec.get(target_img, kpss1)
            self.feature_image.append([box,target_representation])
            distances = []
            for index, instance in df.iterrows():
                source_representation = instance[f"{os.path.basename(self.model_path)}_representation"]
                sim = self.rec.compute_sim(target_representation, source_representation)
                distances.append(sim)
            
            threshold = 0.43

            result_df[f"distance_{os.path.basename(self.model_path)}"] = distances
            result_df = result_df.drop(columns=[f"{os.path.basename(self.model_path)}_representation"])
            result_df = result_df[result_df[f"distance_{os.path.basename(self.model_path)}"] > threshold]
            result_df = result_df.sort_values(
                by=[f"distance_{os.path.basename(self.model_path)}"], ascending=True
            ).reset_index(drop=True)
            return result_df
        except Exception as e:
            print(f"[analyze_video_insightface][recognition]: {e}")
            return []
        
    def recognition_age_gender(self, list_face_image):
        faces_input = prepare_classification_images(list_face_image,device=self.age_gender_model.device)
        output = self.age_gender_model.inference(faces_input)
        list_age_gender = []
        age_output = output[:, 2]
        gender_output = output[:, :2].softmax(-1)
        gender_probs, gender_indx = gender_output.topk(1)
        age, gender = None, None
        for index in range(output.shape[0]):
            age = age_output[index].item()
            age = age * (self.age_gender_model.meta.max_age - self.age_gender_model.meta.min_age) + self.age_gender_model.meta.avg_age
            age = round(age, 2)

            if gender_probs is not None:
                gender_score = gender_probs[index].item()
                gender = "male" if gender_indx[index].item() == 0 else "female"

            list_age_gender.append([age, gender])
        return list_age_gender

    def welcomespeech(self):
        try:
            while True:
                if self.is_running_greeting == False:
                    break
                index_label = self.buffer_label.get()
                if index_label == "":
                    continue
                label = self.list_labels[index_label]
                if label == '':
                    return
                else:
                    print("[welcome speech] Label: ",label)
                hour = datetime.now().hour
                # label = checklabel2voice(label)
                if label == 'unknown' or label == 'None': return
                if 0<=hour<12:
                    # self.text2voice_gtts(f'Chào buổi sáng {label}')
                    self.play_sound(f'Chào ngày mới {label}')
                elif 12<=hour<17:
                    # self.text2voice_gtts(f'Chào buổi chiều {label}')
                    self.play_sound(f'Chào buổi chiều {label}')
                elif 17<=hour<=23:
                    # self.text2voice_gtts(f'Chào buổi tối {label}')
                    self.play_sound(f'Hẹn gặp lại hôm sau {label}')
        except Exception as e:
            print(f"[welcomespeech]: {e}")

    def text2voice_gtts(self,text):
        try:
            language = 'vi'
            myobj = gTTS(text=text, lang=language, slow=False)
            # Saving the converted audio in a mp3 file named
            text = unidecode(text.replace(" ", "_"))
            path_sound = f"t2v_{text}.wav"
            if not os.path.exists(path_sound):
                myobj.save(path_sound)
            try:
                print(f"[text2voice_gtts] Path: {path_sound}")
                self.play_audio(path_sound)
                print(f"[text2voice_gtts]: {text}")
            except Exception as e: 
                print(f"[text2voice_gtts]: {e}")
            
            # if os.path.exists(path_sound):
            #     # Delete the file
            #     os.remove(path_sound)
        except Exception as e:
            print(f"[text2voice_gtts]: {e}")
    def play_sound(self,text, lang='vi'):
        try:
            tts = gTTS(text=text, lang=lang)
            tts.save(f'temp_{self.video_path}.mp3')
            print(f"Save sound: {text}")
            pygame.mixer.init()
            pygame.mixer.music.load(f'temp_{self.video_path}.mp3')
            pygame.mixer.music.play()
            pygame.time.wait(5000)
            # Clean up
            pygame.mixer.quit()
        except Exception as e:
            print(f"[play_sound]: {e}")

    def recognition_insightface_frame(self, frame, box, kpss1):
        try:
            label = ""
            position = ""
            if box[0] > 0 and box[1] > 0 and box[2] < frame.shape[1] and box[3] < frame.shape[0] and box[3] > 0:
                dfs = self.recognition(frame,box, kpss1)
            else:
                dfs = pd.DataFrame()
            if dfs.shape[0] > 0:
                candidate = dfs.iloc[0]
                label = candidate["labels"] 
                position = candidate["position"]
            else:
                label = None
                position = ""
            
            self.list_label_greeting = list(filter(lambda x: x not in self.labels_to_remove, self.list_label_greeting))
            self.labels_to_remove = []

            if label in self.list_label_disable.keys():
                self.list_label_disable[label] = 0
            
            if self.list_label_greeting.count(label) < 3:
                self.list_label_greeting.append(label)
            if self.list_label_greeting.count(label) ==3:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Add buffer : {label}")
                indices = [i for i, x in enumerate(self.list_labels) if x == label]
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Indices: {indices}")
                if len(indices) > 0:
                    self.buffer_label.put(indices[0])
                self.list_label_greeting.append(label)
                
            else:
                self.buffer_label.put("")

            bbox = [max(0, int(x)) if i < 4 else x for i, x in enumerate(box[:5])]
                
            instance = [bbox,label, kpss1, position]
            return instance, frame
        except Exception as e:
            print(f"[analyze_video_insightface][recognition_insightface_frame]: {e}")
            return []
        
    def analyze_detect_face(self, frame):
        try:
            list_instance = []
            bboxes, kpss = self.detector.autodetect(frame, max_num=5)
            if len(bboxes) > 0:
                for i,box in enumerate(bboxes):
                    instance, frame = self.recognition_insightface_frame(frame, box, kpss[i])
                    list_instance.append(instance)
            return list_instance
        except Exception as e:
            print(f"[analyze_video_insightface][analyze_detect_face]: {e}")
            return []
        
    def get_feature(self, frame):
        list_feature = []
        bboxes, kpss = self.detector.autodetect(frame, max_num=5)
        if len(bboxes) > 0:
            for i,box in enumerate(bboxes):
                feature = self.rec.get(frame, kpss[i])
                list_feature.append(feature)
        return list_feature


    def analyze_insightface_frame(self, frame):
        list_instance = []
        kpss1 = []
        list_instance_update = []
        self.feature_image = []
        try:
            list_face_image = []
            list_box_age_gender = []
            list_label_mask = []
            bboxes1, kpss1 = self.detector.autodetect(frame, max_num=20)
            self.list_label_disable = {key: value + 1 for key, value in self.list_label_disable.items()}
            self.labels_to_remove = [label for label, value in self.list_label_disable.items() if value >= 30]
            if len(bboxes1) > 0:
                for i,box in enumerate(bboxes1):
                    instance, frame = self.recognition_insightface_frame(frame, box, kpss1[i])
                    list_instance.append(instance)

                    x, y, w, h = int(box[0]), int(box[1]), int(box[2]-box[0]), int(box[3]-box[1])
                    y1 = int(max(0, y-h/4))
                    x1 = int(max(0, x-w/4))
                    x2 = int(min(frame.shape[1], x+w+w/4))
                    
                    bbox = [max(0, int(x)) if i < 4 else x for i, x in enumerate(box[:5])]
                    face_image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    color = (0, 255, 255)
                    if face_image.shape[0] == 0 or face_image.shape[1] == 0:
                        return [], frame
                    label_mask = 0
                    if face_image.shape[0] > 32 and face_image.shape[1] > 32:
                        label_mask = self.mask_analyze.maskProcess(face_image)
                    else:
                        label_mask = 0
                    list_label_mask.append(label_mask)

                    img = frame[y1:y + h, x1:x2]
                    if img.shape[0] > 0 and img.shape[1] > 0:
                        list_face_image.append(img)
                        list_box_age_gender.append([x, y,w, h])

            if len(list_face_image) > 0:
                list_age_gender = self.recognition_age_gender(list_face_image)
            for i,instance in enumerate(list_instance):
                instance.append(list_age_gender[i][0])
                instance.append(list_age_gender[i][1])
                instance.append(list_label_mask[i])
                list_instance_update.append(instance)    
            return list_instance_update
        except Exception as e:
            print(f"[analyze_video_insightface][analyze_insightface_frame]: {e}")
            return list_instance
    
    def calculate_iou(self,box1, box2):
        try:
            x1, y1, w1, h1 = box1[0], box1[1], box1[2]-box1[0], box1[3]-box1[1]
            x2, y2, w2, h2 = box2[0], box2[1], box2[2]-box2[0], box2[3]-box2[1]

            x1_left, x1_right = x1, x1 + w1
            y1_top, y1_bottom = y1, y1 + h1
            x2_left, x2_right = x2, x2 + w2
            y2_top, y2_bottom = y2, y2 + h2

            x_intersection = max(0, min(x1_right, x2_right) - max(x1_left, x2_left))
            y_intersection = max(0, min(y1_bottom, y2_bottom) - max(y1_top, y2_top))
            intersection_area = x_intersection * y_intersection

            area1 = w1 * h1
            area2 = w2 * h2
            union_area = area1 + area2 - intersection_area

            iou = intersection_area / union_area

            return iou
        except Exception as e:
            print(f"[analyze_video_insightface][calculate_iou]: {e}")
            return 0
        
    def match_face_to_person(self,person_boxes, face_boxes, iou_threshold=0):
        try:
            matched_pairs = {}
            for person_idx, face_box in enumerate(person_boxes):
                best_iou = 0
                best_person_idx = None

                for face_idx, face_box in enumerate(face_boxes):
                    iou = self.calculate_iou(person_boxes[person_idx], face_box[0])
                    if iou > best_iou and iou >= iou_threshold:
                        best_iou = iou
                        best_person_idx = face_idx

                if best_person_idx is not None:
                    matched_pairs[person_idx] = best_person_idx

            matched_persons = [face_boxes[matched_pairs[i]] if i in matched_pairs else None for i in range(len(person_boxes))]

            return matched_persons
        except Exception as e:
            print(f"[analyze_video_insightface][match_face_to_person]: {e}")
            return []
        
    def extract_face(self,frame, face_box):
        try:
            x, y, w, h = face_box
            frame_height, frame_width = frame.shape[:2]
            if w < 64:
                target_size = (128, 128)
            else:
                target_size = (int(w*2), int(h*2))  
            delta_w = target_size[0] - w
            delta_h = target_size[1] - h
            x1, y1 = int(x - delta_w // 2), int(y - delta_h // 2)
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = int(min(frame_width, x1 + target_size[0]))
            y2 = int(min(frame_height, y1 + target_size[1]))
            expanded_face = frame[y1:y2, x1:x2]
            return expanded_face
        except Exception as e:
            print(f"[{datetime.now().strftime(DATETIME_FORMAT)}][analyze_video][extract_face]: ", e)
            return None
    def count_most_frequent_element(self,lst):
        if len(lst) == 0:
            return None
        
        element_count = Counter(lst)
        max_count = max(element_count.values())
        most_frequent_elements = [key for key, value in element_count.items() if value == max_count]
        
        return most_frequent_elements

    def average_number(self,age_list):
        if not age_list or age_list is None:  # Check if the list is empty to avoid division by zero
            return 0
        total_age = sum(age_list)
        average = total_age / len(age_list)
    
        return average

    def analyze_video(self,camera_analyze,image):
        frame = image.copy()
        time_start = time.time()
        for (object_detection, pred_boxes) in self.person_analyze.analyze_tracking_frame(frame):
            object_detection = object_detection
            pred_boxes = pred_boxes
        time_end = time.time()
        # print(f"[{datetime.now().strftime(DATETIME_FORMAT)}]Time tracking: {time_end - time_start}")
        if object_detection is None:
            return image
        
        list_image_label = []
        
        list_recognition_insightface = []
        time_start_recognition = time.time()
        list_recognition_insightface = self.analyze_insightface_frame(frame)
        time_end_recognition = time.time()
        # print(f"[{datetime.now().strftime(DATETIME_FORMAT)}]Time recognition: {time_end_recognition - time_start_recognition}")
        list_person_features = []
        list_recognition_insightface_merge = self.match_face_to_person(pred_boxes, list_recognition_insightface)
        if object_detection is not None:
            if list_recognition_insightface_merge is not None and len(list_recognition_insightface_merge) > 0:
                object_detection.list_labels = list_recognition_insightface_merge
            object_detection.list_person_counting = list_recognition_insightface
            frame, list_person_features = object_detection.plot()
        
            if list_person_features is not None and len(list_person_features) > 0:
                for person_feature in list_person_features:
                    label_name = person_feature[0]
                    label_mask = person_feature[1]
                    age = person_feature[2]
                    gender = person_feature[3]
                    id = person_feature[4]
                    box_face = person_feature[5]
                    box_person = person_feature[6]
                    main_color_clothes = person_feature[7]
                    name_color = person_feature[8]
                    position = person_feature[9]

                    extend_face_image = None
                    person_image = None
                    face_image = None
                    guid = id
                    # if self.index_frame % 200 == 0:
                    #     ref_id = f"{id}_{self.index_frame}"
                    #     guid = ref_id
                    if len(box_face) > 0:
                        face_image = image[box_face[1]:box_face[3]+box_face[1], box_face[0]:box_face[2]+box_face[0]]
                    
                    if face_image is None or len(face_image) == 0:
                        face_image = None
                    else:
                        extend_face_image = self.extract_face(image, box_face)
                    if len(box_person) > 0:
                        person_image = image[box_person[1]:box_person[3]+box_person[1], box_person[0]:box_person[2]+box_person[0]]
                    
                    path_dir_image = ""
                    path_save_face_image = ""
                    path_save_person_image = ""

                    if label_name != "" and extend_face_image is not None:
                        path_dir_image = f"{STATIC_FOLDER}/results/Camera_{camera_analyze.name_video}/{datetime.now().strftime(DATE_TIME_FORMAT)}/id_{label_name.replace(' ', '')}"
                        path_save_face_image = f"{path_dir_image}/{time.time()}.jpg"
                        
                        if person_image is not None:
                            path_save_person_image = f"{path_dir_image}/person_{time.time()}.jpg"
                    elif label_name == "":
                        if face_image is not None or person_image is not None:
                            path_dir_image = f"{STATIC_FOLDER}/results/Camera_{camera_analyze.name_video}/{datetime.now().strftime(DATE_TIME_FORMAT)}/id_{guid}"
                        if face_image is not None:
                            path_save_face_image = f"{path_dir_image}/{time.time()}.jpg"

                        if person_image is not None:
                            path_save_person_image = f"{path_dir_image}/person_{time.time()}.jpg"
                    # path_image = f"{path_dir_image}/origin_{time.time()}.jpg"
                    if path_dir_image != "" and not os.path.exists(path_dir_image):
                        os.makedirs(path_dir_image)
                    elif path_dir_image == "":
                        path_save_face_image = ""
                        path_save_person_image = ""

                    if guid not in camera_analyze.list_total_id:
                        person_model = PersonModel()
                        # person model
                        person_model.id = guid
                        if label_name is not None and label_name != "":
                            person_model.list_face_name.append(label_name)
                            person_model.label_name = label_name
                        if label_mask is not None:
                            person_model.list_check_masks.append(label_mask)
                            person_model.average_check_mask = label_mask
                        
                        if age is not None:
                            person_model.list_age.append(age)
                            person_model.average_age = age
                        if gender is not None:
                            person_model.list_gender.append(gender)
                            person_model.average_gender = gender
                        if main_color_clothes is not None:
                            person_model.code_color = f"{main_color_clothes[0]},{main_color_clothes[1]},{main_color_clothes[2]}"
                        if name_color is not None:
                            person_model.name_color = name_color
                        person_model.list_image_path = []
                        time_label = datetime.now().strftime(DATETIME_FORMAT)
                        if extend_face_image is not None and path_save_face_image != "":
                            if extend_face_image.shape[0] > 0 and person_image.shape[1] > 0:
                                extend_face_image = cv2.resize(extend_face_image, (128, 128))
                                cv2.imwrite(path_save_face_image, extend_face_image)
                            
                                list_image_label.append([path_save_face_image, label_name, position, age, gender, id, main_color_clothes,label_mask, time_label])
                                if path_save_face_image not in person_model.list_image_path:
                                    path_save_face_image.replace("back-end\\","")
                                    person_model.list_image_path.append(path_save_face_image)

                        if person_image is not None and person_image.shape[0] > 0 and person_image.shape[1] > 0:
                            if person_image.shape[1] > 128 and person_image.shape[1] > person_image.shape[0]:
                                person_image = cv2.resize(person_image, (128, int(person_image.shape[0] * 128 / person_image.shape[1])))
                            elif person_image.shape[0] > 128 and person_image.shape[1] <= person_image.shape[0]:
                                person_image = cv2.resize(person_image, (int(person_image.shape[1] * 128 / person_image.shape[0]),128))

                            cv2.imwrite(path_save_person_image, person_image)
                            if path_save_person_image not in person_model.list_image_path:
                                path_save_person_image.replace("back-end\\","")
                                person_model.list_image_path.append(path_save_person_image)
                            list_image_label.append([path_save_person_image, label_name, position, age, gender, id, main_color_clothes,label_mask, time_label])
                        
                        person_model.time = int(time.time())

                        camera_analyze.list_person_model.append(person_model)
                        camera_analyze.list_total_id.append(guid)

                    else:
                        index = camera_analyze.list_total_id.index(guid)
                        if camera_analyze.index_frame % 3 == 0:
                            if age is not None:
                                camera_analyze.list_person_model[index].list_age.append(age)
                                if len(camera_analyze.list_person_model[index].list_age) > 50:
                                    del camera_analyze.list_person_model[index].list_age[0]
                            
                            if gender is not None:
                                camera_analyze.list_person_model[index].list_gender.append(gender)
                                if len(camera_analyze.list_person_model[index].list_gender) > 50:
                                    del camera_analyze.list_person_model[index].list_gender[0] 
                        camera_analyze.list_person_model[index].counting_tracking += 1
                        if label_name != "" and label_name is not None:
                            camera_analyze.list_person_model[index].label_name = label_name

                        if (camera_analyze.list_person_model[index].counting_tracking % 1 == 0 or camera_analyze.list_person_model[index].counting_tracking == 2 or (camera_analyze.list_person_model[index].label_name != label_name and label_name != "")):
                            camera_analyze.list_person_model[index].average_gender = self.count_most_frequent_element(camera_analyze.list_person_model[index].list_gender)
                            camera_analyze.list_person_model[index].average_age = self.average_number(camera_analyze.list_person_model[index].list_age)
                            # camera_analyze.list_person_model[index].average_gender = gender
                            # camera_analyze.list_person_model[index].average_age = age
                            time_label = datetime.now().strftime(DATETIME_FORMAT)
                            if extend_face_image is not None and path_save_face_image != "":
                                if extend_face_image.shape[0] > 0 and person_image.shape[1] > 0:
                                    extend_face_image = cv2.resize(extend_face_image, (128, 128))
                                    cv2.imwrite(path_save_face_image, extend_face_image)
                                    list_image_label.append([path_save_face_image, label_name, position, age, gender, id, main_color_clothes,label_mask, time_label])
                                    camera_analyze.list_person_model[index].list_image_path.append(path_save_face_image)
                            elif person_image is not None and person_image.shape[0] > 0 and person_image.shape[1] > 0:
                                if person_image.shape[1] > 128 and person_image.shape[1] > person_image.shape[0]:
                                    person_image = cv2.resize(person_image, (128, int(person_image.shape[0] * 128 / person_image.shape[1])))
                                elif person_image.shape[0] > 128 and person_image.shape[1] <= person_image.shape[0]:
                                    person_image = cv2.resize(person_image, (int(person_image.shape[1] * 128 / person_image.shape[0]),128))

                                cv2.imwrite(path_save_person_image, person_image)
                                list_image_label.append([path_save_person_image, label_name, position, age, gender, id, main_color_clothes,label_mask, time_label])
                                camera_analyze.list_person_model[index].list_image_path.append(path_save_person_image)
                                       
            # Send data to report
            if camera_analyze.index_frame % 100 == 0 or camera_analyze.index_frame ==camera_analyze.frame_count - 2:
                print(f"[analyze_video][index_frame] {camera_analyze.video_path_report} Frame: {camera_analyze.index_frame}, Number of people: {len(camera_analyze.list_person_model)}")
                if len(camera_analyze.list_person_model) > 0:
                    send_report_to_db(camera_analyze)
            if len(camera_analyze.list_person_model) >= 1000:      
                camera_analyze.list_total_id.remove(camera_analyze.list_total_id[0])
                camera_analyze.list_person_model.remove(camera_analyze.list_person_model[0])
        return frame,list_image_label
    
    def analyze_image(self,camera_analyze = None,image=None, is_report=False):
        frame = image.copy()
        for (object_detection, pred_boxes) in self.person_analyze.analyze_predict_frame(frame):
            object_detection = object_detection
            pred_boxes = pred_boxes
        if object_detection is None:
            return image
        
        list_image_label = []
        list_recognition_insightface = []
        list_recognition_insightface = self.analyze_insightface_frame(frame)
        list_person_features = []
        list_recognition_insightface_merge = self.match_face_to_person(pred_boxes, list_recognition_insightface)
        if object_detection is not None:
            if list_recognition_insightface_merge is not None and len(list_recognition_insightface_merge) > 0:
                object_detection.list_labels = list_recognition_insightface_merge
            object_detection.list_person_counting = list_recognition_insightface
            frame, list_person_features = object_detection.plot()
            if is_report:
                if list_person_features is not None and len(list_person_features) > 0:
                    for person_feature in list_person_features:
                        label_name = person_feature[0]
                        label_mask = person_feature[1]
                        age = person_feature[2]
                        gender = person_feature[3]
                        id = person_feature[4]
                        box_face = person_feature[5]
                        box_person = person_feature[6]
                        main_color_clothes = person_feature[7]
                        name_color = person_feature[8]
                        position = person_feature[9]

                        extend_face_image = None
                        person_image = None
                        face_image = None
                        guid = id
                        if self.index_frame % 200 == 0:
                            ref_id = f"{id}_{self.index_frame}"
                            guid = ref_id
                        if len(box_face) > 0:
                            face_image = image[box_face[1]:box_face[3]+box_face[1], box_face[0]:box_face[2]+box_face[0]]
                        
                        if face_image is None or len(face_image) == 0:
                            face_image = None
                        else:
                            extend_face_image = self.extract_face(image, box_face)
                        if len(box_person) > 0:
                            person_image = image[box_person[1]:box_person[3]+box_person[1], box_person[0]:box_person[2]+box_person[0]]
                        
                        path_dir_image = ""
                        path_save_face_image = ""
                        path_save_person_image = ""

                        if label_name != "" and extend_face_image is not None:
                            path_dir_image = f"{STATIC_FOLDER}/results/Camera_{camera_analyze.name_video}/{datetime.now().strftime(DATE_TIME_FORMAT)}/id_{label_name.replace(' ', '')}"
                            path_save_face_image = f"{path_dir_image}/{time.time()}.jpg"
                            
                            if person_image is not None:
                                path_save_person_image = f"{path_dir_image}/person_{time.time()}.jpg"
                        elif label_name == "":
                            if face_image is not None or person_image is not None:
                                path_dir_image = f"{STATIC_FOLDER}/results/Camera_{camera_analyze.name_video}/{datetime.now().strftime(DATE_TIME_FORMAT)}/id_{guid}"
                            if face_image is not None:
                                path_save_face_image = f"{path_dir_image}/{time.time()}.jpg"

                            if person_image is not None:
                                path_save_person_image = f"{path_dir_image}/person_{time.time()}.jpg"
                        path_image = f"{path_dir_image}/origin_{time.time()}.jpg"
                        if path_dir_image != "" and not os.path.exists(path_dir_image):
                            os.makedirs(path_dir_image)
                        elif path_dir_image == "":
                            path_save_face_image = ""
                            path_save_person_image = ""

                        person_model = PersonModel()
                        # person model
                        person_model.id = guid
                        if label_name is not None and label_name != "":
                            person_model.list_face_name.append(label_name)
                            person_model.label_name = label_name
                        if label_mask is not None:
                            person_model.list_check_masks.append(label_mask)
                            person_model.average_check_mask = label_mask
                        
                        if age is not None:
                            person_model.list_age.append(age)
                            person_model.average_age = age
                        if gender is not None:
                            person_model.list_gender.append(gender)
                            person_model.average_gender = gender
                        if main_color_clothes is not None:
                            person_model.code_color = f"{main_color_clothes[0]},{main_color_clothes[1]},{main_color_clothes[2]}"
                        if name_color is not None:
                            person_model.name_color = name_color
                        person_model.list_image_path = []
                        time_label = datetime.now().strftime(DATETIME_FORMAT)
                        if extend_face_image is not None and path_save_face_image != "":
                            if extend_face_image.shape[0] > 0 and person_image.shape[1] > 0:
                                extend_face_image = cv2.resize(extend_face_image, (128, 128))
                                cv2.imwrite(path_save_face_image, extend_face_image)
                                list_image_label.append([path_save_face_image, label_name, position, age, gender, id, main_color_clothes,label_mask, time_label])
                                if path_save_face_image not in person_model.list_image_path:
                                    path_save_face_image.replace("back-end\\","")
                                    person_model.list_image_path.append(path_save_face_image)

                        elif person_image is not None and person_image.shape[0] > 0 and person_image.shape[1] > 0:
                            person_image = cv2.resize(person_image, (int(person_image.shape[1] * 128 / person_image.shape[0]), 128))
                            cv2.imwrite(path_save_person_image, person_image)
                            if path_save_person_image not in person_model.list_image_path:
                                path_save_person_image.replace("back-end\\","")
                                person_model.list_image_path.append(path_save_person_image)
                            list_image_label.append([path_save_person_image, label_name, position, age, gender, id, main_color_clothes,label_mask, time_label])
                        
                        cv2.imwrite(path_image, frame)
                        person_model.time = int(time.time())
                        camera_analyze.list_person_model.append(person_model)
                print(f"[analyze_video][index_frame] {camera_analyze.video_path_report} Frame: {camera_analyze.index_frame}, Number of people: {len(camera_analyze.list_person_model)}")
                if len(camera_analyze.list_person_model) > 0:
                    send_report_to_db(camera_analyze)
        return frame,list_image_label
    
class FaceAnalysisInsightFaceThread(QThread):
     def __init__(self, worker, path=""):
          super(FaceAnalysisInsightFaceThread, self).__init__()
          self.worker = worker
          self.path_video = path

     def run(self):
          self.worker.face_analyzer = FaceAnalysisInsightFace(self.path_video)
        

        

            
