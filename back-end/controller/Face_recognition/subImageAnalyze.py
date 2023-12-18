import os
import threading
import time
from server.reports.services import add_video_service
import cv2
from controller.Face_recognition.analyze_video_insightface import FaceAnalysisInsightFace
from PySide2.QtGui import (QPixmap,QImage)
from PySide2.QtWidgets import *
from config import STATIC_FOLDER
from PySide2.QtCore import QRunnable, Signal, QObject
import numpy as np

from PySide2.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt,QThreadPool)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter,QPixmap,QImage,
    QRadialGradient)

class CameraWorkerSignals(QObject):
    result = Signal(np.ndarray)
    finished = Signal()
    updateUI = Signal(list) 

class SubImageAnalyze(QRunnable):
    def __init__(self,video_path, face_analyzer):
        super().__init__()
        self.video_path = video_path
        self.video_path_report = video_path
        self.signals = CameraWorkerSignals()

        self.is_running = True

        self.index_frame = 0

        self.list_person_model = []
        self.list_total_id = []
        self.list_id_check_in_frame = []
        self.list_image_label = []

        self.face_analyzer = face_analyzer
    def run(self):
            self.image = cv2.imread(self.video_path)

            self.width_frame = self.image.shape[1]
            self.height_frame = self.image.shape[0]
            if type(self.video_path) == str:
                self.name_video = os.path.basename(str(self.video_path))
                path_dir = f"{STATIC_FOLDER}\\{os.path.basename(str(self.video_path))}"
                if not os.path.exists(path_dir):
                    os.makedirs(path_dir)
                self.output_image = f"{path_dir}\\output.jpg"

            img0 = self.image.copy()
            self.index_frame += 1
            self.face_analyzer.index_frame = self.index_frame   
            image, list_image_label = self.face_analyzer.analyze_image(self,img0,True)
            print(list_image_label)
           
            self.list_image_label = list_image_label
            # self.output_video.write(img0)
            self.signals.result.emit(image)
            self.signals.updateUI.emit(self.list_image_label)

    def stop(self):
        self.face_analyzer.is_running_greeting = False
        self.face_analyzer.buffer_label.put("")

        self.is_running = False
        list_thread = threading.enumerate()
        print(f"Active thread names: {', '.join([thread.name for thread in list_thread])}")
                
class CameraWidget(QWidget):
    def __init__(self, path, index, parent=None,analyzer=None):
        super(CameraWidget, self).__init__(parent)
        self.path = path
        self.thread_pool = parent.thread_pool
        self.analyzer =  analyzer
        
        self.list_camera = parent.list_camera
        self.list_camera_screen = parent.list_camera_screen
        self.grid_layout = parent.grid_layout

        self.camera_layout = QVBoxLayout()

        self.camera_label = QLabel(self)
        self.close_button = QPushButton(self)

        self.close_button.setObjectName(u"close_button_{}".format(index))
        self.close_button.setStyleSheet(u"QPushButton {\n"
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
        self.close_button.setIcon(QIcon(u":/16x16/icons/16x16/cil-x.png"))
        self.close_button.setIconSize(QSize(16, 16))
        self.close_button.setFixedSize(30, 30)

        self.camera_layout.addWidget(self.camera_label)
        self.camera_layout.addWidget(self.close_button)

        self.camera_label.setObjectName(u"camera_label_{}".format(index))
        self.camera_label.setStyleSheet("border: 2px solid red;")
        self.camera_label.setText("Loading...")

        self.close_button.clicked.connect(self.stop_camera)
        self.camera_label.setAlignment(Qt.AlignCenter)

        self.setLayout(self.camera_layout)
        self.start_camera()

    def start_camera(self):
        add_video_service(self.path)
        self.worker = SubImageAnalyze(self.path, self.analyzer)
        self.worker.signals.result.connect(self.display_image)
        self.worker.signals.finished.connect(self.thread_pool.waitForDone)
        # Execute the worker in a separate thread from the thread pool
        self.thread_pool.start(self.worker)

    def stop_camera(self):
        # Stop the camera capture
        if self.worker:
            self.worker.signals.result.disconnect(self.display_image)
            time.sleep(1)
            self.worker.stop()
        self.thread_pool.clear()
        print(f"Length thread pool: {self.thread_pool.activeThreadCount()}")
        self.camera_layout.removeWidget(self.camera_label)
        self.camera_label.deleteLater()
        self.camera_layout.removeWidget(self.close_button)
        self.close_button.deleteLater()
        self.grid_layout.removeWidget(self.list_camera_screen[self.path])
        self.list_camera_screen[self.path].deleteLater()
        del self.camera_label
        del self.close_button
        del self.list_camera_screen[self.path]
        index = self.list_camera.index(str(self.path))
        self.list_camera.remove(str(self.path))
        self.update_camera_layout_positions(index)

    def update_camera_layout_positions(self, removed_index):
        if len(self.list_camera_screen) > 4:
            self.num_col = 3
        elif len(self.list_camera_screen) == 1:
            self.num_col = 1
        else:
            self.num_col = 2
        for i, path in enumerate(self.list_camera):
            if i >= removed_index:
                # Calculate the new row and column for the remaining layouts
                row = i // self.num_col
                col = i % self.num_col

                # Update the position of the layout in the grid_layout
                self.grid_layout.removeWidget(self.list_camera_screen[path])
                self.grid_layout.addWidget(self.list_camera_screen[path], row, col, 1, 1)
                self.list_camera_screen[path].setObjectName(u"camera_label_{}".format(i + 1))

    def display_image(self, frame):
        # Display the captured frame
        if frame is None:
            self.camera_label.setText("No frame")
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.camera_label.size(), Qt.KeepAspectRatio)
        # check camera_label attribute camerawidget
        if self.camera_label:
            self.camera_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        if self.camera_label:
            self.camera_label.setFixedSize(self.size())
        super().resizeEvent(event)

