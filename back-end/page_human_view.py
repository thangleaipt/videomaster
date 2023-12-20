
import math
import os
import threading
from PySide2.QtCore import (QSize,QThreadPool)
from PySide2.QtGui import (QIcon)

from PySide2.QtWidgets import *
from server.reports.services import add_video_service
from controller.Face_recognition.analyze_video_insightface import FaceAnalysisInsightFace
from controller.Face_recognition.subHumanAnalyze import CameraWidget
import cv2
import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


segment_count = 4

class MyDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thêm video")
        self.setMinimumSize(QSize(300, 150))
        self.setMaximumSize(QSize(1000, 500))


        # Tạo ô văn bản (QLineEdit)
        self.text_edit = QLineEdit(self)
        self.text_edit.setStyleSheet(u"QLineEdit {\n"
        "	background-color: rgb(27, 29, 35);\n"
        "	border-radius: 5px;\n"
        "	border: 2px solid rgb(27, 29, 35);\n"
        "	padding-left: 10px;\n"
        "	color: rgb(255, 255, 255);\n"
        "}\n"
        "QLineEdit:hover {\n"
        "	border: 2px solid rgb(64, 71, 88);\n"
        "}\n"
        "QLineEdit:focus {\n"
        "	border: 2px solid rgb(91, 101, 124);\n"
        "}")
        self.text_edit.setPlaceholderText("Nhập tên camera")

        self.ok_button = QPushButton('Save', self)
        self.ok_button.clicked.connect(self.accept)
        self.ok_button.setStyleSheet(
            "QPushButton {"
            "	border: 2px solid rgb(27, 29, 35);"
            "	border-radius: 5px;	"
            "	background-color: rgb(27, 29, 35);"
            "	color: white;"
            "}"
            "QPushButton:hover {"
            "	background-color: rgb(57, 65, 80);"
            "	border: 2px solid rgb(61, 70, 86);"
            "}"
            "QPushButton:pressed {"
            "	background-color: rgb(35, 40, 49);"
            "	border: 2px solid rgb(43, 50, 61);"
            "}"
        )
        self.ok_button.setFixedHeight(30)


        # Tạo layout cho dialog
        layout = QVBoxLayout(self)
        layout.addWidget(self.text_edit)
        layout.addWidget(self.ok_button)

        # Áp dụng stylesheet cho QDialog (background và border)
        self.setStyleSheet(
            "QDialog {"
            "background-color: transparent;"
            "}"
        )

class PAGEHUMAN(QWidget):
    def __init__(self):
        super().__init__()
        self.new_size = 0
        self.list_camera_screen = {}
        self.file_path = ""
        self.scroll_area = None
        self.list_camera = []
        self.thread_pool = QThreadPool()
        self.set_ui()
        self.setObjectName(u"page_human")
    def set_ui(self):
        self.main_layout = QHBoxLayout(self)
        self.control_layout = QVBoxLayout()
        self.add_button_layout = QHBoxLayout()
        self.grid_layout = QGridLayout()
        self.grid_layout.setObjectName(u"gridLayout")
        self.list_camera_labels = {}
        self.list_camera_layout = {}
        self.list_close_button = {}
       
        # self.scroll_area = QScrollArea()
        # self.scroll_area.setWidgetResizable(True)
        # scroll_content = QWidget()
        # self.ref_layout = QVBoxLayout(scroll_content)
         # Set the content widget for the scroll area
        # self.scroll_area.setWidget(scroll_content)
    
        # Add video button
        self.add_video_button = QPushButton("Add Video")
        self.add_video_button.setStyleSheet(u"QPushButton {\n"
                "	border: 2px solid rgb(27, 29, 35);\n"
                "	border-radius: 5px;	\n"
                "	background-color: rgb(27, 29, 35);\n"
                "}\n"
                "QPushButton:hover {\n"
                "	background-color: rgb(57, 65, 80);\n"
                "	border: 2px solid rgb(61, 70, 86);\n"
                "}\n"
                "QPushButton:pressed {	\n"
                "	background-color: rgb(35, 40, 49);\n"
                "	border: 2px solid rgb(43, 50, 61);\n"
                "}")
        icon3 = QIcon()
        icon3.addFile(u":/16x16/icons/16x16/cil-folder-open.png", QSize(), QIcon.Normal, QIcon.Off)
        self.add_video_button.setIcon(icon3)
        self.add_video_button.clicked.connect(self.show_dialog_video)
        self.add_button_layout.addWidget(self.add_video_button)

        # Thêm grid_layout và QScrollArea vào main_layout
        self.control_layout.addLayout(self.grid_layout)
        self.control_layout.addLayout(self.add_button_layout)
        self.main_layout.addLayout( self.control_layout)
        # self.main_layout.addWidget(self.scroll_area)
        # loading_screen = LoadingScreen(self.thread_pool)
        # loading_screen.show()
        self.init_camera()

    def init_camera(self):
        if segment_count >4 :
            num_col = 3
        elif segment_count == 1:
            num_col = 1
        else:
            num_col = 2
        self.thread_pool.setMaxThreadCount(segment_count)
        for i in range(segment_count):
            camera_widget = CameraWidget(i + 1,self)
            self.grid_layout.addWidget(camera_widget, i // num_col, i % num_col, 1, 1)
            self.list_camera_screen[i] = camera_widget

    def resizeEvent(self, event):
        # Override the resizeEvent to handle window resize
        self.new_size = event.size()
        self.add_video_button.setFixedSize(int(self.new_size.width()/10* 2), 50)
        # Call the base class implementation
        super().resizeEvent(event)

    def split_video(self,input_path):
            list_video_split = []
            video = cv2.VideoCapture(input_path)

            dir_folder = os.path.dirname(input_path)
            name_video = os.path.basename(input_path).split(".")[0]
            output_folder = os.path.join(dir_folder, f"output_{name_video}")
            
            if not video.isOpened():
                    raise Exception("Không thể mở file video.")

            clip = mp.VideoFileClip(input_path)
            clip_duration = clip.duration
            
            segment_duration = int(clip_duration/4)
            
            remainder = clip_duration % segment_duration
            os.makedirs(output_folder, exist_ok=True)
            for i in range(segment_count):
                    start_time = i * segment_duration
                    last_segment_duration = segment_duration if i < segment_count - 1 else remainder+segment_duration
                    end_time = start_time + last_segment_duration
                    output_path = os.path.join(output_folder, f"segment_{i+1}.mp4")
                    if not os.path.exists(output_path):    
                            ffmpeg_extract_subclip(input_path, start_time, end_time, targetname=output_path)
                    list_video_split.append(output_path)
            video.release()
            return list_video_split

            
    def show_dialog_video(self):
        # if len(self.list_camera_screen) > 0:
        #     for path_camera in reversed(self.list_camera):
        #          self.list_camera_screen[path_camera].stop_camera()
        self.list_camera = [] 

        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Video files (*.mp4 *.avi *.mov)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setViewMode(QFileDialog.Detail)
        list_video_split = []
        if file_dialog.exec_():
            self.file_path = file_dialog.selectedFiles()
            if self.file_path and self.file_path != "":
                add_video_service(self.file_path[0])
                print("Selected file:", self.file_path[0])
                list_video_split = self.split_video(self.file_path[0])
                # list_video_split.append(self.file_path[0])
                for index, path in enumerate(list_video_split):
                    self.list_camera_screen[index].path = path
                    self.list_camera_screen[index].path_origin = self.file_path[0]
                    self.list_camera_screen[index].start_camera()
                    self.list_camera.append(str(index))
                
       

            