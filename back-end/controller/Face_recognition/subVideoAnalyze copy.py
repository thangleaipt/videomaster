import os
import threading
import time
from server.reports.services import add_video_service
import cv2
from controller.Face_recognition.analyze_video_insightface import FaceAnalysisInsightFace
from PySide2.QtGui import (QPixmap,QImage)
from PySide2.QtWidgets import *
from PyQt5.QtCore import QTimer
from config import STATIC_FOLDER
from PySide2.QtCore import QThread

class SubVideoAnalyze(QThread):
    def __init__(self,video_path,camera_label, ref_layout):
        super().__init__()
        self.video_path = video_path
        self.camera_label = camera_label
        self.is_facerecog_model = True
        self.ref_layout = ref_layout

        self.face_analyzer = FaceAnalysisInsightFace()
        self.index_frame = 0
        self.list_person_model = []
        self.list_total_id = []
        self.list_id_check_in_frame = []
        self.list_image_label = []
        if self.video_path.isdigit():
            self.video_path = int(self.video_path)
        self.video_capture = cv2.VideoCapture(self.video_path)
        self.fps = self.video_capture.get(cv2.CAP_PROP_FPS)
        self.codec = self.video_capture.get(cv2.CAP_PROP_FOURCC)
        self.height_frame = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width_frame = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if type(self.video_path) == str:
            if 'rtsp' in self.video_path:
                path_dir = f"{STATIC_FOLDER}\\{os.path.dirname(str(self.video_path)).replace(':', '').replace('/', '')}"
                self.name_video = os.path.dirname(str(self.video_path)).replace(':', '').replace('/', '')
            else:
                self.name_video = os.path.basename(str(self.video_path))
                path_dir = f"{STATIC_FOLDER}\\{os.path.basename(str(self.video_path))}"
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            self.output_video = cv2.VideoWriter(f"{path_dir}\\output.mp4",cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.frame_width, self.frame_height))
        else:
            self.name_video = self.video_path
            path_dir = f"{STATIC_FOLDER}\\{str(self.name_video)}"
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            self.output_video = cv2.VideoWriter(f"{path_dir}\\output.mp4",cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.frame_width, self.frame_height))
        

    def viewCam(self):
        if self.video_capture is not None and self.video_capture.isOpened():
            ret, frame = self.video_capture.read()
            list_image_label = []
            if ret:
                img0 = frame.copy()
                self.index_frame += 1
                self.face_analyzer.index_frame = self.index_frame   
                time_start_recognition = time.time()
                img0, list_image_label = self.face_analyzer.analyze_video(self,img0)
                time_end_recognition = time.time()
                # print(f"Number of active threads: {threading.active_count()}")
                active_threads = threading.enumerate()
                thread_names = [thread.name for thread in active_threads]
                print(f"Active thread names: {', '.join(thread_names)}")
                # print(f"Time recognition [{self.video_path}]: {time_end_recognition - time_start_recognition}")
    
                self.list_image_label = list_image_label
                self.update_ui()
                # self.output_video.write(img0)
                image = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (self.width_frame, self.height_frame))
                height, width, channel = image.shape
                step = channel * self.width_frame
                # create QImage from image
                qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
                # show image in img_label
                self.camera_label.setPixmap(QPixmap.fromImage(qImg))
            else:
                if not isinstance(self.video_path, int) and 'rtsp' not in self.video_path:
                    self.video_capture.release()
                    self.video_capture = None
                    # self.output_video.release()
                    print(f"{self.video_path} DONE")
                    self.camera_label.setPixmap(QPixmap())
                    return
                else:
                    self.video_capture.release()
                    self.video_capture = cv2.VideoCapture(self.video_path)
                    self.index_frame = 0
                    print(f"{self.video_path} RESTART")

    def update_ui(self):
        try:
            for i in range(len(self.list_image_label)):
                self.h_layout = QHBoxLayout()  # Create a QHBoxLayout for each row
                self.recognition_image = cv2.imread(self.list_image_label[i][0])
                self.recognition_image = cv2.cvtColor(self.recognition_image, cv2.COLOR_BGR2RGB)
                self.recognite_image = QImage(self.recognition_image, self.recognition_image.shape[1], self.recognition_image.shape[0], self.recognition_image.shape[2]*self.recognition_image.shape[1], QImage.Format_RGB888)
                self.image_label = QLabel(f"Label {i * 2 + 1}")

                if self.list_image_label[i][1] is not None and self.list_image_label[i][1] != "":
                    self.image_label.setStyleSheet("border: 1px solid green; border-radius: 5px; margin-bottom: 5px;")
                else:
                    self.image_label.setStyleSheet("border: 1px solid red; border-radius: 5px; margin-bottom: 5px;")
                self.image_label.setPixmap(QPixmap.fromImage(self.recognite_image))
                # image_label.setScaledContents(True)
                self.text_label = QLabel(f"Label {i * 2 + 2}")
                self.text_label.setStyleSheet("border: 1px solid gray; border-radius: 5px; margin-bottom: 5px;")
            
                if len(self.list_image_label[i]) > 1:
                    self.text_label.setText(self.update_text(self.list_image_label[i]))
                else:
                    self.text_label.setText(f"")
                self.h_layout.addWidget(self.image_label)
                self.h_layout.addWidget(self.text_label)
                self.ref_layout.insertLayout(0, self.h_layout)
        except Exception as e:
            print(f"[update_ui]: {e}")
    
    def update_text(self,label):
        label_text = f"<b>Họ và tên: {label[1]}</b>"
        label_text += f"<br>Bộ phận: {label[2]}"
        label_text += f"<br>Tuổi: {label[3]}"
        label_text += f"<br>Giới tính: {label[4]}"
        if label[7] == 1:
            label_text += f"<br>Đeo khẩu trang"
        label_text += f"<br>Thời gian: {label[8]}"
        
        return label_text
                
    def stop(self):
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
            self.output_video.release()
            self.camera_label.setPixmap(QPixmap())
            self.isRunning = False
            self.terminate()
            self.deleteLater()
            self.wait(1000)
            

    def run(self):
        self.timer = QTimer()

        if not self.timer.isActive():
            add_video_service(self.video_path)
            self.timer.start(1000 // self.fps)
        else:
            print("stop timer")
            self.timer.stop()
            self.video_capture.release()

        self.timer.timeout.connect(self.viewCam)


    

