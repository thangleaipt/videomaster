
import math
import os
import threading
from PySide2.QtCore import (QSize,QThreadPool)
from PySide2.QtGui import (QIcon)

from PySide2.QtWidgets import *
from controller.Face_recognition.subImageAnalyze import CameraWidget


class PAGEIMAGE(QWidget):
    def __init__(self):
        super().__init__()
        self.new_size = 0
        self.list_camera_screen = {}
        self.scroll_area = None
        self.list_camera = []
        self.thread_pool = QThreadPool()
        self.set_ui()
        self.setObjectName(u"page_image")
        self.analyzer = None
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
    
        # Add video button
        self.add_image_button = QPushButton("Add Image")
        self.add_image_button.setStyleSheet(u"QPushButton {\n"
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
        self.add_image_button.setIcon(icon3)
        self.add_image_button.clicked.connect(self.show_dialog_image)
        self.add_button_layout.addWidget(self.add_image_button)

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
                camera_widget = CameraWidget(path, i + 1, self, self.analyzer)
                self.grid_layout.addWidget(camera_widget, i // num_col, i % num_col, 1, 1)
                self.list_camera_screen[path] = camera_widget
    
  
    def resizeEvent(self, event):
        # Override the resizeEvent to handle window resize
        self.new_size = event.size()
        self.scroll_area.setFixedSize(int(self.new_size.width()/10* 2), self.new_size.height())
        self.add_image_button.setFixedSize(int(self.new_size.width()/10* 2), 50)

        # Call the base class implementation
        super().resizeEvent(event)

    def show_dialog_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image files (*.jpeg *.jpg *.png)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setViewMode(QFileDialog.Detail)

        if file_dialog.exec_():
            file_path = file_dialog.selectedFiles()
            if file_path:
                print("Selected file:", file_path[0])
                self.list_camera.append(file_path[0])
                self.init_camera()
       

            