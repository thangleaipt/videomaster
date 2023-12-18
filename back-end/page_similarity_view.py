
import math
import os
import threading
from PySide2.QtCore import (QSize,QThreadPool)
from PySide2.QtGui import (QIcon)

from PySide2.QtWidgets import *
import cv2
import moviepy.editor as mp
from PySide2.QtGui import (QPixmap,QImage)


class PAGESIMILARITY(QWidget):
    def __init__(self):
        super().__init__()
        self.new_size = 0
        self.list_camera_screen = {}
        self.list_features = []
        self.list_camera = []
        self.thread_pool = QThreadPool()
        self.set_ui()
        self.setObjectName(u"page_similarity")
        self.analyzer = None
    def set_ui(self):
        self.main_layout = QHBoxLayout(self)
        self.control_layout = QVBoxLayout()
        self.add_button_layout = QHBoxLayout()
        self.grid_layout = QGridLayout()
        self.grid_layout.setObjectName(u"gridLayout")
        self.list_camera_layout = {}
        self.list_close_button = {}
    
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

        self.delete_image_button = QPushButton("Delete Image")
        icon3 = QIcon()
        icon3.addFile(u":/16x16/icons/16x16/cil-remove.png", QSize(), QIcon.Normal, QIcon.Off)
        self.delete_image_button.setIcon(icon3)
        self.delete_image_button.clicked.connect(self.delete_image)
        self.add_button_layout.addWidget(self.delete_image_button)
        # disable delete button
        if len(self.list_camera) == 0:
            self.delete_image_button.setEnabled(False)
            self.delete_image_button.setStyleSheet(u"QPushButton {\n"
                "	border: 2px solid rgb(57, 65, 80);\n"
                "	border-radius: 5px;	\n"
                "	background-color: rgb(57, 65, 80);\n"
                "}\n")

        # Thêm grid_layout và QScrollArea vào main_layout
        self.control_layout.addLayout(self.grid_layout)
        self.control_layout.addLayout(self.add_button_layout)
        self.main_layout.addLayout( self.control_layout)

    # close image label
    def delete_image(self):
        for key in self.list_camera_screen.keys():
            self.grid_layout.removeWidget(self.list_camera_screen[key])
            self.list_camera_screen[key].deleteLater()

        self.list_camera_screen = {}
        self.list_camera = []
        self.list_features = []    

    def init_camera(self):
        num_col = 2
        self.thread_pool.setMaxThreadCount(len(self.list_camera))
        for i,path in enumerate(self.list_camera):
            if path not in self.list_camera_screen.keys():
                image = cv2.imread(path)
                frame, list_image_label = self.analyzer.analyze_image(None,image,False)
                feature_image = self.analyzer.feature_image
                self.list_features.append(feature_image)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

                pixmap = QPixmap.fromImage(q_image)
                self.image_label_1 = QLabel()
                self.image_label_1.setScaledContents(True)
                self.image_label_1.setPixmap(QPixmap(pixmap))
                self.grid_layout.addWidget(self.image_label_1, i // num_col, i % num_col, 1, 1)
                self.list_camera_screen[path] = self.image_label_1

        max_similarity = 0
        if len(self.list_camera) == 2:
            for feature in self.list_features[1]:
                similarity = self.analyzer.rec.compute_sim(self.list_features[0][0][1], feature[1])
                if max_similarity < similarity:
                    max_similarity = similarity
                    box = feature[0]
            if max_similarity < 0.45:
                QMessageBox.warning(self, "Warning", "Two images are not similar enough") 
                self.list_camera_screen[self.list_camera[1]].setPixmap(QPixmap(""))
            else:
                persentage = max_similarity * 100
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                cv2.putText(image, f"{persentage:.2f}%", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                q_image_update = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap_update = QPixmap.fromImage(q_image_update)
                self.list_camera_screen[self.list_camera[1]].setPixmap(QPixmap(pixmap_update))
                print(f"Distance 2 image: {persentage}")
  
    def resizeEvent(self, event):
        # Override the resizeEvent to handle window resize
        self.new_size = event.size()
        self.add_image_button.setFixedSize(int(self.new_size.width()/10* 2), 50)
        self.delete_image_button.setFixedSize(int(self.new_size.width()/10* 2), 50)

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
                if len(self.list_camera) > 0:
                    self.delete_image_button.setEnabled(True)
                    self.delete_image_button.setStyleSheet(u"QPushButton {\n"
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
                self.init_camera()
       

            