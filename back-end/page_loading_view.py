
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
import cv2
import numpy as np
import unidecode

from ui_loading_screen import Ui_LoadingScreen
from widgets_loading import CircularProgress
from PIL import Image
from ultralytics import YOLO
from controller.boxmot.trackers.strongsort.strong_sort import ReIDDetectMultiBackend
from pathlib import Path
import torch
import os

from config import WEIGHT_FOLDER

column_ratios = [0.1, 0.15, 0.1, 0.1,0.1,0.15,0.15,0.15]
device = torch.device(0)
reid = ReIDDetectMultiBackend(
    weights=Path(os.path.join(WEIGHT_FOLDER,'osnet_ain_x1_0_msmt17.pt')),
    device=device
)
date_time_format = "yyyy-MM-dd hh:mm:ss"
model = YOLO('models/yolov8m.pt')
model.to(device)
# Check torch cuda
print(f"Is CUDA available: {torch.cuda.is_available()}")

class FilterThread(QThread):
    finished = Signal()
    progress_update = Signal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report = parent

    def run(self):
        self.fill_report()
        self.finished.emit()
    
    def fill_report(self):
                if len(self.report.list_reports_filter) >= 16:
                        self.report.tableWidget.setRowCount(len(self.report.list_reports_filter))
                else:
                        self.report.tableWidget.setRowCount(16)
                self.report.tableWidget.clearContents()

                screen_width = QDesktopWidget().screenGeometry().width()
                column_widths = [int(ratio * screen_width) for ratio in column_ratios]
                for i in range(8):
                        self.report.tableWidget.setColumnWidth(i, column_widths[i])
                for i, report in enumerate(self.report.list_reports_filter):
                        print(f"Counter: {self.report.counter}")
                        self.progress_update.emit(self.report.counter)
                        self.report.counter = round(i / len(self.report.list_reports_filter) * 100)
                        self.report.tableWidget.setItem(i, 0, QTableWidgetItem(str(i)))
                        if 'random' in report['person_name']:
                                name = "Người không xác định"
                        else:
                                name = report['person_name']
                        self.report.tableWidget.setItem(i, 1, QTableWidgetItem(str(name)))
                        self.report.tableWidget.setItem(i, 2, QTableWidgetItem(str(report['age'])))
                        if report['gender'] == 1:
                                gender = "Nam"
                        elif report['gender'] == 0:
                                gender = "Nữ"
                        else:
                                gender = "Không xác định"
                        self.report.tableWidget.setItem(i, 3, QTableWidgetItem(str(gender)))
                        if report['mask'] == 1:
                                mask = "Có"
                        elif report['mask'] == 0:
                                mask = "Không"
                        self.report.tableWidget.setItem(i, 4, QTableWidgetItem(str(mask)))
                        if report['code_color'] is None:
                                color = "Không xác định"
                                self.report.tableWidget.setItem(i, 5, QTableWidgetItem(str(color)))
                        else:
                                color = report['code_color']
                                numbers = [int(num) for num in color.split(',')]
                                image_color = Image.new('RGB', (128, 128), (numbers[0], numbers[1], numbers[2]))
                                image_pil = image_color.tobytes()
                                q_image = QImage(image_pil, image_color.width, image_color.height, QImage.Format_RGB888)

                                # Tạo QPixmap từ QImage
                                pixmap_color = QPixmap.fromImage(q_image)
                                item = QTableWidgetItem()
                                item.setData(Qt.DecorationRole, pixmap_color)
                                self.report.tableWidget.setItem(i, 5, item)
                        
                        time_start = QDateTime.fromString(self.report.time, date_time_format)
                        time_target = time_start.addSecs(int(int(report['time'])/5))
                        time_string = time_target.toString(date_time_format)
                        self.report.tableWidget.setItem(i, 6, QTableWidgetItem(str(time_string)))
                        if len(report['images']) > 0:
                                image_path = report['images'][0]['path']
                                pixmap = QPixmap(image_path).scaledToWidth(128, Qt.SmoothTransformation).scaledToHeight(128, Qt.SmoothTransformation)
                                item = QTableWidgetItem()
                                item.setData(Qt.DecorationRole, pixmap)
                                self.report.tableWidget.setItem(i, 7, item)

                                self.report.tableWidget.setRowHeight(i, pixmap.height())

                                self.report.tableWidget.setColumnWidth(4, pixmap.width() + 20)

class ImportThread(QThread):
    finished = Signal()
    progress_update = Signal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report = parent
        self.counter = 0

    def run(self):
        self.filter_report_query()
        self.fill_report()
        self.finished.emit()
        
    def filter_report_query(self):
                number = 0
                for file_path in self.report.list_file_path:
                        print("Selected file:", file_path)
                        frame_import = cv2.imread(file_path)
                        max_report = None
                        min_report = None
                        self.report.list_reports_filter = []
                        list_instance = self.report.analyzer.analyze_detect_face(frame_import)

                        if len(list_instance) > 0 and list_instance[0][1] is not None:
                                for report in self.report.list_reports:
                                        number += 1
                                        self.counter = round(number / len(self.report.list_reports) * 100)
                                        if len(self.report.list_reports) <number:
                                               self.counter = 100
                                        self.progress_update.emit(self.counter)
                                        if unidecode(report['person_name']).lower() == unidecode(list_instance[0][1]).lower():
                                                self.report.list_reports_filter.append(report)
                        # Unknown person and have face
                        elif len(list_instance) > 0 and list_instance[0][1] is None: 
                                feature_image_import = self.report.analyzer.get_feature(frame_import)[0]
                                for report in self.report.list_reports:
                                        number += 1
                                        self.counter = round(number / len(self.report.list_reports) * 100)
                                        if len(self.report.list_reports) <number:
                                               self.counter = 100
                                        self.progress_update.emit(self.counter)
                                        list_class_image = report['images']
                                        list_path_face_image = []
                                        for image in list_class_image:
                                                name_image = os.path.basename(image['path'])
                                                if "face_" in name_image:
                                                        list_path_face_image.append(image['path'])
                                               
                                        for path_image in list_path_face_image:
                                                frame_ref = cv2.imread(path_image)
                                                feature_ref = self.report.analyzer.get_feature(frame_ref)
                                                if len(feature_ref) > 0:
                                                        similarity = self.report.analyzer.rec.compute_sim(feature_image_import, feature_ref[0])
                                                else:
                                                        similarity = 0
                                                if similarity > 0.43:
                                                        max_report = report
                                                        break
                                                
                                        if max_report is not None and max_report not in self.report.list_reports_filter:             
                                                self.report.list_reports_filter.append(max_report)
                                                print("Max report: ", max_report['person_name'])
                        if len(list_instance) == 0 or len(self.report.list_reports_filter) >= 0:
                                h_import,w1_import,_ = frame_import.shape
                                xyxys_import =  np.array([[0,0,w1_import,h_import]])
                                feature_image_import = reid.get_features(xyxys_import,frame_import)[0]
                                for report in self.report.list_reports:
                                        if report not in self.report.list_reports_filter:
                                                number += 1
                                                self.counter = round(number / len(self.report.list_reports) * 100)
                                                if len(self.report.list_reports) <number:
                                                        self.counter = 100
                                                self.progress_update.emit(self.counter)
                                                list_path_person_image = []
                                                list_class_image = report['images']
                                                for image in list_class_image:
                                                        name_image = os.path.basename(image['path'])
                                                        if "person_" in name_image:
                                                                list_path_person_image.append(image['path'])
                                
                                                for path_image in list_path_person_image:
                                                        frame_ref = cv2.imread(path_image)
                                                        h_ref,w1_ref,_ = frame_ref.shape
                                                        xyxys_ref =  np.array([[0,0,w1_ref,h_ref]])
                                                        feature_ref = reid.get_features(xyxys_ref,frame_ref)[0]
                                                        dist = self.report._cosine_distance(np.array([feature_image_import]), np.array([feature_ref]))[0][0]
                                                        if dist < 0.17:
                                                                min_report = report
                                                                
                                                if min_report is not None and min_report not in self.report.list_reports_filter:             
                                                        self.report.list_reports_filter.append(min_report)
                                                        print("Min report: ", min_report['person_name'])

    def fill_report(self):
                if len(self.report.list_reports_filter) >= 16:
                        self.report.tableWidget.setRowCount(len(self.report.list_reports_filter))
                else:
                        self.report.tableWidget.setRowCount(16)
                self.report.tableWidget.clearContents()

                screen_width = QDesktopWidget().screenGeometry().width()
                column_widths = [int(ratio * screen_width) for ratio in column_ratios]
                for i in range(8):
                        self.report.tableWidget.setColumnWidth(i, column_widths[i])
                for i, report in enumerate(self.report.list_reports_filter):
                        self.report.tableWidget.setItem(i, 0, QTableWidgetItem(str(i)))
                        if 'random' in report['person_name']:
                                name = "Người không xác định"
                        else:
                                name = report['person_name']
                        self.report.tableWidget.setItem(i, 1, QTableWidgetItem(str(name)))
                        self.report.tableWidget.setItem(i, 2, QTableWidgetItem(str(report['age'])))
                        if report['gender'] == 1:
                                gender = "Nam"
                        elif report['gender'] == 0:
                                gender = "Nữ"
                        else:
                                gender = "Không xác định"
                        self.report.tableWidget.setItem(i, 3, QTableWidgetItem(str(gender)))
                        if report['mask'] == 1:
                                mask = "Có"
                        elif report['mask'] == 0:
                                mask = "Không"
                        self.report.tableWidget.setItem(i, 4, QTableWidgetItem(str(mask)))
                        if report['code_color'] is None:
                                color = "Không xác định"
                                self.report.tableWidget.setItem(i, 5, QTableWidgetItem(str(color)))
                        else:
                                color = report['code_color']
                                numbers = [int(num) for num in color.split(',')]
                                image_color = Image.new('RGB', (128, 128), (numbers[0], numbers[1], numbers[2]))
                                image_pil = image_color.tobytes()
                                q_image = QImage(image_pil, image_color.width, image_color.height, QImage.Format_RGB888)

                                # Tạo QPixmap từ QImage
                                pixmap_color = QPixmap.fromImage(q_image)
                                item = QTableWidgetItem()
                                item.setData(Qt.DecorationRole, pixmap_color)
                                self.report.tableWidget.setItem(i, 5, item)
                        
                        time_start = QDateTime.fromString(self.report.time, date_time_format)
                        time_target = time_start.addSecs(int(int(report['time'])/5))
                        time_string = time_target.toString(date_time_format)
                        self.report.tableWidget.setItem(i, 6, QTableWidgetItem(str(time_string)))
                        if len(report['images']) > 0:
                                image_path = report['images'][0]['path']

                                pixmap = QPixmap(image_path).scaledToWidth(128, Qt.SmoothTransformation).scaledToHeight(128, Qt.SmoothTransformation)
                                item = QTableWidgetItem()
                                item.setData(Qt.DecorationRole, pixmap)
                                self.report.tableWidget.setItem(i, 7, item)

                                self.report.tableWidget.setRowHeight(i, pixmap.height())

                                self.report.tableWidget.setColumnWidth(4, pixmap.width() + 20)


class LoadingScreen(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report = parent
        self.ui = Ui_LoadingScreen()
        self.ui.setupUi(self)
        self.setWindowFlag(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.progress = CircularProgress()
        self.progress.width = 300
        self.progress.height = 300
        self.progress.value = 0
        self.progress.move(25, 25)
        self.progress.setFixedSize(self.progress.width, self.progress.height)
        self.progress.add_shadow(True)
        self.progress.font_size = 16
        self.progress.setParent(self.ui.centralwidget)
        self.progress.show()
        self.show()

    def filter_loading(self):
        self.loading_thread = FilterThread(self.report)
        self.loading_thread.finished.connect(self.loading_finished)
        self.loading_thread.progress_update.connect(self.update_progress)
        self.loading_thread.start()
    
    def import_loading(self):
        self.loading_thread = ImportThread(self.report)
        self.loading_thread.finished.connect(self.loading_finished)
        self.loading_thread.progress_update.connect(self.update_progress)
        self.loading_thread.start()

    def update_progress(self, value):
        self.progress.set_value(value)

    def loading_finished(self):
        self.close()
        self.report.show()

