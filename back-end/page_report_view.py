from datetime import datetime, timezone, timedelta
import math
import os
import threading
import time
from PySide2.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt, QDateTime, QTime)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter,QPixmap,QImage,
    QRadialGradient)

from PySide2.QtWidgets import *
from server.reports.services import get_reports_db
import cv2
from PyQt5.QtCore import pyqtSlot
from unidecode import unidecode
import numpy as np
from controller.boxmot.trackers.strongsort.strong_sort import ReIDDetectMultiBackend
from pathlib import Path
import torch
from config import WEIGHT_FOLDER, STATIC_FOLDER
from PySide2.QtMultimedia import QMediaPlayer, QMediaContent
from PySide2.QtMultimediaWidgets import QVideoWidget
from PIL import Image
from ultralytics import YOLO


column_ratios = [0.1, 0.15, 0.1, 0.1,0.1,0.15,0.15,0.15]
date_time_format = "yyyy-MM-dd hh:mm:ss"
reid = ReIDDetectMultiBackend(
    weights=Path(os.path.join(WEIGHT_FOLDER,'osnet_ain_x1_0_msmt17.pt')),
    device=torch.device(0)
)

device = torch.device(0)

model = YOLO('models/yolov8m.pt')
model.to(device)


class PAGEIMAGEVIEW(QDialog):
        def __init__(self, list_image_path=None, video_id=None, parent=None):
                super().__init__(parent)

                self.list_path_images = list_image_path
                self.video_id = video_id

                path_video_dir = f"{STATIC_FOLDER}/videos/{time.strftime('%Y%m%d')}"
                if not os.path.exists(path_video_dir):
                        os.makedirs(path_video_dir)
                path_video = f"{path_video_dir}/{self.video_id}.mp4"
                if not os.path.exists(path_video):
                        self.create_video_from_images(self.list_path_images, video_name=path_video)

                self.setWindowTitle("Video Dialog")
                self.setGeometry(200, 200, 1280, 720)

                # Video widget
                self.video_widget = QVideoWidget(self)

                # Media player
                self.media_player = QMediaPlayer(self)
                self.media_player.setVideoOutput(self.video_widget)

                # Layout setup
                layout = QVBoxLayout()
                layout.addWidget(self.video_widget)

                self.setLayout(layout)

                # Load a sample video (replace with your own path)
                video_url = QUrl.fromLocalFile(path_video)
                content = QMediaContent(video_url)
                self.media_player.setMedia(content)
                self.play_video()

        def play_video(self):
                self.media_player.play()

        def stop_video(self):
                self.media_player.stop()

        def create_video_from_images(self,image_paths, video_name='video_output.avi', output_size=(256, 256), frame_rate=10):
                if len(image_paths) == 0:
                        QMessageBox.warning(self, "Không có hình ảnh", "Không có hình ảnh", QMessageBox.Ok)
                        return
                img = cv2.imread(image_paths[0])
                height, width = img.shape[:2]

                video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'XVID'), frame_rate, (width, height))

                for image_path in image_paths:
                        img = cv2.imread(image_path)
                        output_image = np.zeros((height, width, 3), dtype=np.uint8)
                        output_image[0:img.shape[0], 0:img.shape[1], :] = img
                        video.write(output_image)

                cv2.destroyAllWindows()
                video.release()
        
class PAGEREPORT(QDialog):
        def __init__(self, index, time, analyzer):
                super().__init__()
                self.list_reports_filter = []
                self.video_id = index
                self.time = time
                self.analyzer = analyzer
                self.setMinimumSize(QSize(800, 600))
                self.set_ui()
                self.retranslateUi()
                self.setWindowTitle(f"{index}")

        def set_ui(self):
                self.verticalLayout_6 = QVBoxLayout(self)
                self.verticalLayout_6.setObjectName(u"verticalLayout_6")

                # Create a group box for the filter controls
                self.filter_groupbox = QGroupBox("Lọc dữ liệu")
                self.filter_groupbox.setObjectName(u"filter_groupbox")

                # Create a layout for the filter group box
                self.filter_layout = QHBoxLayout(self.filter_groupbox)
                self.filter_layout.setObjectName(u"filter_layout")

                self.date_time_layout = QHBoxLayout()

                # Add date-time edit controls to the filter layout
                start_label = QLabel("Start Date:", self.filter_groupbox)
                self.date_time_layout.addWidget(start_label)
                self.date_time_layout.addSpacing(2)
                self.dateTimeEdit_start = QDateTimeEdit(self.filter_groupbox)
                self.dateTimeEdit_start.setObjectName(u"dateTimeEdit_start")
                self.dateTimeEdit_start.setFixedSize(200, 50)
                self.dateTimeEdit_start.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
                start_date = QDateTime.fromString(self.time, date_time_format)
                start_date.setTime(QTime(0, 0, 0))
                self.dateTimeEdit_start.setDateTime(start_date)
                self.date_time_layout.addWidget(self.dateTimeEdit_start)
                self.date_time_layout.addSpacing(20)

                end_label = QLabel("End Date:", self.filter_groupbox)
                self.date_time_layout.addWidget(end_label)
                self.date_time_layout.addSpacing(2)
                self.dateTimeEdit_end = QDateTimeEdit(self.filter_groupbox)
                self.dateTimeEdit_end.setObjectName(u"dateTimeEdit_end")
                self.dateTimeEdit_end.setFixedSize(200, 50)
                self.dateTimeEdit_end.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
                end_date = QDateTime.fromString(self.time, date_time_format)
                end_date.setTime(QTime(23, 59, 59))
                self.dateTimeEdit_end.setDateTime(end_date)
                self.date_time_layout.addWidget(self.dateTimeEdit_end)

                self.filter_layout.addLayout(self.date_time_layout)
                self.filter_layout.addSpacing(50)

                # self.name_layout = QHBoxLayout(self)
                # name_label = QLabel("Name:", self.filter_groupbox)
                # self.name_layout.addWidget(name_label)
                # self.name_layout.addSpacing(2)
                # self.name_box = QLineEdit(self.filter_groupbox)
                # self.name_box.setObjectName(u"name_box")
                # self.name_box.setFixedSize(100, 50)
                # self.name_layout.addWidget(self.name_box)
                # self.filter_layout.addLayout(self.name_layout)
                # self.filter_layout.addSpacing(50)

                # Add other filter controls (e.g., gender combo box, name box) to the filter layout
                self.gender_layout = QHBoxLayout()
                gender_label = QLabel("Giới tính:", self.filter_groupbox)
                self.gender_layout.addWidget(gender_label)
                self.gender_layout.addSpacing(2)
                self.gender_combobox = QComboBox(self.filter_groupbox)
                self.gender_combobox.setObjectName(u"gender_combobox")
                self.gender_combobox.setFixedSize(100, 50)
                self.gender_combobox.addItems(["Tất cả", "Nam", "Nữ"])
                self.gender_layout.addWidget(self.gender_combobox)
                self.filter_layout.addLayout(self.gender_layout)
                self.filter_layout.addSpacing(50)

                self.age_layout = QHBoxLayout()
                name_label = QLabel("Tuổi:", self.filter_groupbox)
                self.age_layout.addWidget(name_label)
                self.age_layout.addSpacing(2)
                self.age_combobox = QComboBox(self.filter_groupbox)
                self.age_combobox.setObjectName(u"age_box")
                self.age_combobox.setFixedSize(100, 50)
                self.age_combobox.addItems(["Tất cả","0-20", "21-40", "41-60", "61-80", "81-100"])
                self.age_layout.addWidget(self.age_combobox)
                self.filter_layout.addLayout(self.age_layout)
                self.filter_layout.addSpacing(50)

                self.mask_layout = QHBoxLayout()
                mask_label = QLabel("Đeo khẩu trang:", self.filter_groupbox)
                self.mask_layout.addWidget(mask_label)
                self.mask_layout.addSpacing(2)
                self.mask_combobox = QComboBox(self.filter_groupbox)
                self.mask_combobox.setObjectName(u"mask_box")
                self.mask_combobox.setFixedSize(100, 50)
                self.mask_combobox.addItems(["Tất cả", "Đeo khẩu trang", "Không đeo khẩu trang"])
                self.mask_layout.addWidget(self.mask_combobox)
                self.filter_layout.addLayout(self.mask_layout)
                self.filter_layout.addSpacing(50)

                # Buton Search
                self.search_button = QPushButton("Search", self.filter_groupbox)
                self.search_button.setObjectName(u"search_button")
                self.search_button.setFixedSize(100, 50)
                self.filter_layout.addWidget(self.search_button)
                self.search_button.setStyleSheet(u"QPushButton {\n"
                "	border: 2px solid rgb(27, 29, 35);\n"
                "	border-radius: 5px;	\n"
                # "	background-color: rgb(27, 29, 35);\n"
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
                icon3.addFile(u":/16x16/icons/16x16/cil-magnifying-glass.png", QSize(), QIcon.Normal, QIcon.Off)
                self.search_button.setIcon(icon3)
                self.search_button.clicked.connect(self.get_list_report)

                 # Buton Search
                self.import_button = QPushButton("Import", self.filter_groupbox)
                self.import_button.setObjectName(u"import_button")
                self.import_button.setFixedSize(100, 50)
                self.filter_layout.addWidget(self.import_button)
                self.import_button.setStyleSheet(u"QPushButton {\n"
                "	border: 2px solid rgb(27, 29, 35);\n"
                "	border-radius: 5px;	\n"
                # "	background-color: rgb(27, 29, 35);\n"
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
                icon3.addFile(u":/16x16/icons/16x16/cil-magnifying-glass.png", QSize(), QIcon.Normal, QIcon.Off)
                self.import_button.setIcon(icon3)
                self.import_button.clicked.connect(self.import_image_query)


                # Add the filter group box to the main layout
                self.verticalLayout_6.addWidget(self.filter_groupbox)


                self.frame_3 = QFrame()
                self.frame_3.setObjectName(u"frame_3")
                self.frame_3.setMinimumSize(QSize(0, 150))
                self.frame_3.setFrameShape(QFrame.StyledPanel)
                self.frame_3.setFrameShadow(QFrame.Raised)
                self.horizontalLayout_12 = QHBoxLayout(self.frame_3)
                self.horizontalLayout_12.setSpacing(0)
                self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
                self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
                self.tableWidget = QTableWidget(self.frame_3)
                if (self.tableWidget.columnCount() < 8):
                        self.tableWidget.setColumnCount(8)
                        __qtablewidgetitem = QTableWidgetItem()
                        self.tableWidget.setHorizontalHeaderItem(0, __qtablewidgetitem)
                        __qtablewidgetitem1 = QTableWidgetItem()
                        self.tableWidget.setHorizontalHeaderItem(1, __qtablewidgetitem1)
                        __qtablewidgetitem2 = QTableWidgetItem()
                        self.tableWidget.setHorizontalHeaderItem(2, __qtablewidgetitem2)
                        __qtablewidgetitem3 = QTableWidgetItem()
                        self.tableWidget.setHorizontalHeaderItem(3, __qtablewidgetitem3)
                        __qtablewidgetitem4 = QTableWidgetItem()
                        self.tableWidget.setHorizontalHeaderItem(4, __qtablewidgetitem4)
                        __qtablewidgetitem5 = QTableWidgetItem()
                        self.tableWidget.setHorizontalHeaderItem(5, __qtablewidgetitem5)
                        __qtablewidgetitem6 = QTableWidgetItem()
                        self.tableWidget.setHorizontalHeaderItem(6, __qtablewidgetitem6)
                        __qtablewidgetitem7 = QTableWidgetItem()
                        self.tableWidget.setHorizontalHeaderItem(7, __qtablewidgetitem7)

                if (self.tableWidget.rowCount() < 16):
                        self.tableWidget.setRowCount(16)
                        font2 = QFont()
                        font2.setFamily(u"Segoe UI")
                        self.tableWidget.setObjectName(u"tableWidget")
                        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                        sizePolicy.setHorizontalStretch(0)
                        sizePolicy.setVerticalStretch(0)
                        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
                        self.tableWidget.setSizePolicy(sizePolicy)
                        palette1 = QPalette()
                        brush6 = QBrush(QColor(210, 210, 210, 255))
                        brush6.setStyle(Qt.SolidPattern)
                        palette1.setBrush(QPalette.Active, QPalette.WindowText, brush6)
                        brush15 = QBrush(QColor(39, 44, 54, 255))
                        brush15.setStyle(Qt.SolidPattern)
                        palette1.setBrush(QPalette.Active, QPalette.Button, brush15)
                        palette1.setBrush(QPalette.Active, QPalette.Text, brush6)
                        palette1.setBrush(QPalette.Active, QPalette.ButtonText, brush6)
                        palette1.setBrush(QPalette.Active, QPalette.Base, brush15)
                        palette1.setBrush(QPalette.Active, QPalette.Window, brush15)
                        brush16 = QBrush(QColor(210, 210, 210, 128))
                        brush16.setStyle(Qt.NoBrush)
                #if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
                        palette1.setBrush(QPalette.Active, QPalette.PlaceholderText, brush16)
                #endif
                        palette1.setBrush(QPalette.Inactive, QPalette.WindowText, brush6)
                        palette1.setBrush(QPalette.Inactive, QPalette.Button, brush15)
                        palette1.setBrush(QPalette.Inactive, QPalette.Text, brush6)
                        palette1.setBrush(QPalette.Inactive, QPalette.ButtonText, brush6)
                        palette1.setBrush(QPalette.Inactive, QPalette.Base, brush15)
                        palette1.setBrush(QPalette.Inactive, QPalette.Window, brush15)
                        brush17 = QBrush(QColor(210, 210, 210, 128))
                        brush17.setStyle(Qt.NoBrush)
                #if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
                        palette1.setBrush(QPalette.Inactive, QPalette.PlaceholderText, brush17)
                #endif
                        palette1.setBrush(QPalette.Disabled, QPalette.WindowText, brush6)
                        palette1.setBrush(QPalette.Disabled, QPalette.Button, brush15)
                        palette1.setBrush(QPalette.Disabled, QPalette.Text, brush6)
                        palette1.setBrush(QPalette.Disabled, QPalette.ButtonText, brush6)
                        palette1.setBrush(QPalette.Disabled, QPalette.Base, brush15)
                        palette1.setBrush(QPalette.Disabled, QPalette.Window, brush15)
                        brush18 = QBrush(QColor(210, 210, 210, 128))
                        brush18.setStyle(Qt.NoBrush)
                #if QT_VERSION >= QT_VERSION_CHECK(5, 12, 0)
                        palette1.setBrush(QPalette.Disabled, QPalette.PlaceholderText, brush18)
                #endif
                        self.tableWidget.setPalette(palette1)
                        self.tableWidget.setStyleSheet(u"QTableWidget {	\n"
                "	background-color: rgb(39, 44, 54);\n"
                "	padding: 10px;\n"
                "	border-radius: 5px;\n"
                "	gridline-color: rgb(44, 49, 60);\n"
                "	border-bottom: 1px solid rgb(44, 49, 60);\n"
                "}\n"
                "QTableWidget::item{\n"
                "	border-color: rgb(44, 49, 60);\n"
                "	padding-left: 5px;\n"
                "	padding-right: 5px;\n"
                "	gridline-color: rgb(44, 49, 60);\n"
                "}\n"
                "QTableWidget::item:selected{\n"
                "	background-color: rgb(85, 170, 255);\n"
                "}\n"
                "QScrollBar:horizontal {\n"
                "    border: none;\n"
                "    background: rgb(52, 59, 72);\n"
                "    height: 14px;\n"
                "    margin: 0px 21px 0 21px;\n"
                "	border-radius: 0px;\n"
                "}\n"
                " QScrollBar:vertical {\n"
                "	border: none;\n"
                "    background: rgb(52, 59, 72);\n"
                "    width: 14px;\n"
                "    margin: 21px 0 21px 0;\n"
                "	border-radius: 0px;\n"
                " }\n"
                "QHeaderView::section{\n"
                "	Background-color: rgb(39, 44, 54);\n"
                "	max-width: 30px;\n"
                "	border: 1px solid rgb(44, 49, 60);\n"
                "	border-style: none;\n"
                "    border-bottom: 1px solid rgb(44, 49, 60);\n"
                "    border-right: 1px solid rgb(44, 49, 60);\n"
                "}\n"
                ""
                                        "QTableWidget::horizontalHeader {	\n"
                "	background-color: rgb(81, 255, 0);\n"
                "}\n"
                "QHeaderView::section:horizontal\n"
                "{\n"
                "    border: 1px solid rgb(32, 34, 42);\n"
                "	background-color: rgb(27, 29, 35);\n"
                "	padding: 3px;\n"
                "	border-top-left-radius: 7px;\n"
                "    border-top-right-radius: 7px;\n"
                "}\n"
                "QHeaderView::section:vertical\n"
                "{\n"
                "    border: 1px solid rgb(44, 49, 60);\n"
                "}\n"
                "")
                        self.tableWidget.setFrameShape(QFrame.NoFrame)
                        self.tableWidget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
                        self.tableWidget.setSizeAdjustPolicy(QAbstractScrollArea.AdjustToContents)
                        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
                        self.tableWidget.setAlternatingRowColors(False)
                        self.tableWidget.setSelectionMode(QAbstractItemView.SingleSelection)
                        self.tableWidget.setSelectionBehavior(QAbstractItemView.SelectRows)
                        self.tableWidget.setShowGrid(True)
                        self.tableWidget.setGridStyle(Qt.SolidLine)
                        self.tableWidget.setSortingEnabled(False)
                        self.tableWidget.horizontalHeader().setVisible(True)
                        self.tableWidget.horizontalHeader().setCascadingSectionResizes(True)
                        self.tableWidget.horizontalHeader().setDefaultSectionSize(200)
                        self.tableWidget.horizontalHeader().setStretchLastSection(True)
                        self.tableWidget.verticalHeader().setVisible(False)
                        self.tableWidget.verticalHeader().setCascadingSectionResizes(False)
                        self.tableWidget.verticalHeader().setHighlightSections(False)
                        self.tableWidget.verticalHeader().setStretchLastSection(True)
                        self.horizontalLayout_12.addWidget(self.tableWidget)
                        self.verticalLayout_6.addWidget(self.frame_3)

        def retranslateUi(self):
                        ___qtablewidgetitem = self.tableWidget.horizontalHeaderItem(0)
                        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"STT", None));
                        ___qtablewidgetitem1 = self.tableWidget.horizontalHeaderItem(1)
                        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"Tên", None));
                        ___qtablewidgetitem2 = self.tableWidget.horizontalHeaderItem(2)
                        ___qtablewidgetitem2.setText(QCoreApplication.translate("MainWindow", u"Tuổi", None));
                        ___qtablewidgetitem3 = self.tableWidget.horizontalHeaderItem(3)
                        ___qtablewidgetitem3.setText(QCoreApplication.translate("MainWindow", u"Giới tính", None));
                        ___qtablewidgetitem4 = self.tableWidget.horizontalHeaderItem(4)
                        ___qtablewidgetitem4.setText(QCoreApplication.translate("MainWindow", u"Đeo khẩu trang", None));
                        ___qtablewidgetitem5 = self.tableWidget.horizontalHeaderItem(5)
                        ___qtablewidgetitem5.setText(QCoreApplication.translate("MainWindow", u"Màu Trang Phục", None));
                        ___qtablewidgetitem6 = self.tableWidget.horizontalHeaderItem(6)
                        ___qtablewidgetitem6.setText(QCoreApplication.translate("MainWindow", u"Thời gian nhận diện", None));
                        ___qtablewidgetitem7 = self.tableWidget.horizontalHeaderItem(7)
                        ___qtablewidgetitem7.setText(QCoreApplication.translate("MainWindow", u"Ảnh", None));

                        self.get_list_report()

        def get_list_report(self):
                # Get the Unix timestamp from self.dateTimeEdit_start
                start_timestamp = self.dateTimeEdit_start.dateTime().toSecsSinceEpoch()
                # Get the Unix timestamp from self.dateTimeEdit_end
                end_timestamp = self.dateTimeEdit_end.dateTime().toSecsSinceEpoch()
                page_num = None
                page_size = None

                gender_text = self.gender_combobox.currentText()
                if gender_text == "Nam":
                        gender = 1
                elif gender_text == "Nữ":
                        gender = 0
                else:
                        gender = None
                mask_text = self.mask_combobox.currentText()
                if mask_text == "Đeo khẩu trang":
                        mask = 1
                elif mask_text == "Không đeo khẩu trang":
                        mask = 0
                else:
                        mask = None       
                age = self.age_combobox.currentText()
                if age == "Tất cả":
                        begin_age = 0
                        end_age = 100
                else:
                        begin_age = int(age.split("-")[0])
                        end_age = int(age.split("-")[1])
                print(f"begin_age: {begin_age} end_age: {end_age}")
                self.list_reports = get_reports_db(self.video_id, page_num, page_size, start_timestamp, end_timestamp, begin_age, end_age, gender, mask)
                self.list_reports_filter = self.list_reports
                self.fill_report()

        def convert_timestamp_to_datetime(self,timestamp):
                dt_utc = datetime.utcfromtimestamp(timestamp)
                
                dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                
                dt_vietnam = dt_utc.astimezone(timezone(timedelta(hours=7)))
                
                dt_vietnam_str = dt_vietnam.strftime('%Y-%m-%d %H:%M:%S')
                
                return dt_vietnam_str

        def fill_report(self):
                if len(self.list_reports) >= 16:
                        self.tableWidget.setRowCount(len(self.list_reports_filter))
                else:
                        self.tableWidget.setRowCount(16)
                self.tableWidget.clearContents()

                screen_width = QDesktopWidget().screenGeometry().width()
                column_widths = [int(ratio * screen_width) for ratio in column_ratios]
                for i in range(8):
                        self.tableWidget.setColumnWidth(i, column_widths[i])

                for i, report in enumerate(self.list_reports_filter):
                        self.tableWidget.setItem(i, 0, QTableWidgetItem(str(i)))
                        if 'random' in report['person_name']:
                                name = "Người lạ"
                        else:
                                name = report['person_name']
                        self.tableWidget.setItem(i, 1, QTableWidgetItem(str(name)))
                        self.tableWidget.setItem(i, 2, QTableWidgetItem(str(report['age'])))
                        if report['gender'] == 1:
                                gender = "Nam"
                        elif report['gender'] == 0:
                                gender = "Nữ"
                        else:
                                gender = "Không xác định"
                        self.tableWidget.setItem(i, 3, QTableWidgetItem(str(gender)))
                        if report['mask'] == 1:
                               mask = "Có"
                        elif report['mask'] == 0:
                                mask = "Không"
                        self.tableWidget.setItem(i, 4, QTableWidgetItem(str(mask)))
                        if report['code_color'] is None:
                                color = "Không xác định"
                                self.tableWidget.setItem(i, 5, QTableWidgetItem(str(color)))
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
                                self.tableWidget.setItem(i, 5, item)
                        
                        self.tableWidget.setItem(i, 6, QTableWidgetItem(str(self.convert_timestamp_to_datetime(report['time']))))
                        if len(report['images']) > 0:
                                image_path = report['images'][0]['path']

                                pixmap = QPixmap(image_path).scaledToWidth(128, Qt.SmoothTransformation).scaledToHeight(128, Qt.SmoothTransformation)
                                item = QTableWidgetItem()
                                item.setData(Qt.DecorationRole, pixmap)
                                self.tableWidget.setItem(i, 7, item)

                                self.tableWidget.setRowHeight(i, pixmap.height())

                                self.tableWidget.setColumnWidth(4, pixmap.width() + 20)

                self.tableWidget.cellClicked.connect(self.on_row_selected)

        def on_row_selected(self):
                selected_rows = self.tableWidget.selectionModel().selectedRows()
                list_image_path = []
                if selected_rows:
                        item = [index.row() for index in selected_rows][0]
                        list_image = self.list_reports_filter[item]['images']
                        video_id = self.list_reports_filter[item]['id']
                        for image in list_image:
                                path_image = image['path']
                                list_image_path.append(path_image)
                        page_image = PAGEIMAGEVIEW(list_image_path,video_id)
                        page_image.exec_()

        def _cosine_distance( self,a, b):
                """cosine_distance matrix

                Args:
                        a (array): matrix a
                        b (array): matrix b

                Returns:
                        float: cosine_distance
                """        
                # if not data_is_normalized:
                a = np.asarray(a) / np.linalg.norm(a)
                b = np.asarray(b) / np.linalg.norm(b)
                return 1. - np.dot(a, b.T)
        
        def import_image_query(self):
                file_dialog = QFileDialog()
                file_dialog.setNameFilter("Image files (*.jpeg *.jpg *.png)")
                file_dialog.setFileMode(QFileDialog.ExistingFile)
                file_dialog.setViewMode(QFileDialog.Detail)

                self.list_reports_filter = []
                max_similarity = 0
                max_report = None
                min_report = None

                if file_dialog.exec_():
                        file_path = file_dialog.selectedFiles()
                        if file_path:
                                print("Selected file:", file_path[0])
                                frame_import = cv2.imread(file_path[0])
                                list_instance = self.analyzer.analyze_detect_face(frame_import)

                                if len(list_instance) > 0 and list_instance[0][1] is not None:
                                        for report in self.list_reports:
                                                if unidecode(report['person_name']).lower() == unidecode(list_instance[0][1]).lower():
                                                        self.list_reports_filter.append(report)
                                # Unknown person and have face
                                elif len(list_instance) > 0 and list_instance[0][1] is None: 
                                        feature_image_import = self.analyzer.get_feature(frame_import)[0]
                                        for report in self.list_reports:
                                                list_path_image = []
                                                list_class_image = report['images']
                                                for image in list_class_image:
                                                        list_path_image.append(image['path'])
                                                for path_image in list_path_image:
                                                        frame_ref = cv2.imread(path_image)
                                                        feature_ref = self.analyzer.get_feature(frame_ref)
                                                        if len(feature_ref) > 0:
                                                                similarity = self.analyzer.rec.compute_sim(feature_image_import, feature_ref[0])
                                                        else:
                                                                similarity = 0
                                                        if max_similarity < similarity and similarity > 0.45:
                                                                max_similarity = similarity
                                                                max_report = report
                                                               
                                                print(f"max_similarity: {max_similarity}")

                                                if max_report is not None and max_report not in self.list_reports_filter:             
                                                        self.list_reports_filter.append(max_report)
                                if len(self.list_reports_filter) == 0:
                                        h_import,w1_import,_ = frame_import.shape
                                        xyxys_import =  np.array([[0,0,w1_import,h_import]])
                                        feature_image_import = reid.get_features(xyxys_import,frame_import)[0]
                                        min_similarity = 1
                                        for report in self.list_reports:
                                                list_path_image = []
                                                list_class_image = report['images']
                                                for image in list_class_image:
                                                        list_path_image.append(image['path'])
                                                for path_image in list_path_image:
                                                        frame_ref = cv2.imread(path_image)
                                                        results = model(frame_ref,classes=[0],conf=0.4,verbose=False)
                                                        pred_boxes = results[0].boxes
                                                        for pred in pred_boxes:
                                                                box = pred.xyxy.squeeze().tolist()
                                                                xyxys_ref =  np.array([[box[0],box[1],box[2],box[3]]])
                                                                feature_ref = reid.get_features(xyxys_ref,frame_ref)[0]
                                                                dist = self._cosine_distance(np.array([feature_image_import]), np.array([feature_ref]))[0][0]
                                                                if min_similarity > dist and dist < 0.2:
                                                                        min_similarity = dist
                                                                        min_report = report
                                                                       
                                                                print(f"distance_similarity: {dist} {image['path']}")
                                                if min_report is not None and min_report not in self.list_reports_filter:             
                                                        self.list_reports_filter.append(min_report)
                                
                                if len(self.list_reports_filter) == 0:
                                        QMessageBox.information(self, "Notification", "Không tìm thấy kết quả.")
                                self.fill_report()
                                                        
