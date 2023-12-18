
import math
import os
import threading
from PySide2.QtCore import (QSize,QThreadPool)

from PySide2.QtWidgets import *
from controller.Face_recognition.subCameraAnalyze import CameraWidget
import cv2
import moviepy.editor as mp
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

class MyDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thêm camera")
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

class PAGECAMERA(QWidget):
    def __init__(self):
        super().__init__()
        self.new_size = 0
        self.list_camera_screen = {}
        self.scroll_area = None
        self.list_camera = []
        self.thread_pool = QThreadPool()
        self.set_ui()
        self.setObjectName(u"page_camera")
    def set_ui(self):
        self.main_layout = QHBoxLayout(self)
        self.control_layout = QVBoxLayout()
        self.add_button_layout = QHBoxLayout()
        self.grid_layout = QGridLayout()
        self.grid_layout.setObjectName(u"gridLayout")
        self.list_camera_labels = {}
        self.list_camera_layout = {}
        self.list_close_button = {}
       
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.ref_layout = QVBoxLayout(scroll_content)
         # Set the content widget for the scroll area
        self.scroll_area.setWidget(scroll_content)
        # Add camera button
        self.add_camera_button = QPushButton("Add Camera")
        self.add_camera_button.setStyleSheet(u"QPushButton {\n"
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
        self.add_camera_button.clicked.connect(self.show_dialog_camera)
        self.add_button_layout.addWidget(self.add_camera_button)

        # Thêm grid_layout và QScrollArea vào main_layout
        self.control_layout.addLayout(self.grid_layout)
        self.control_layout.addLayout(self.add_button_layout)
        self.main_layout.addLayout( self.control_layout)
        self.main_layout.addWidget(self.scroll_area)

    def init_camera(self):
        if len(self.list_camera) >4 :
            num_col = 3
        elif len(self.list_camera) == 1:
            num_col = 1
        else:
            num_col = 2
        self.thread_pool.setMaxThreadCount(len(self.list_camera))
        for i,path in enumerate(self.list_camera):
            if path not in self.list_camera_screen.keys():
                camera_widget = CameraWidget(path, i + 1, self)
                self.grid_layout.addWidget(camera_widget, i // num_col, i % num_col, 1, 1)
                self.list_camera_screen[path] = camera_widget
    
            
    def resizeEvent(self, event):
        # Override the resizeEvent to handle window resize
        self.new_size = event.size()
        self.scroll_area.setFixedSize(int(self.new_size.width()/10* 2), self.new_size.height())
        self.add_camera_button.setFixedSize(int(self.new_size.width()/10* 2), 50)

        # Call the base class implementation
        super().resizeEvent(event)

    
    def show_dialog_camera(self):
        dialog = MyDialog()
        result = dialog.exec_()  
        if result == QDialog.Accepted:
            path_camera = dialog.text_edit.text()
            if path_camera not in self.list_camera:
                self.list_camera.append(path_camera)
                self.init_camera()
            else:
                 QMessageBox.warning(self, "Camera đã được thêm", "Camera đã được thêm", QMessageBox.Ok)
       

            