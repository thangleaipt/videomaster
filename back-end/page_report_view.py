from datetime import datetime, timezone, timedelta
import os
import subprocess
import time
from PySide2.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt, QDateTime, QTime, QTimeZone, QFileInfo)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter,QPixmap,QImage,
    QRadialGradient)

from PySide2.QtWidgets import *
from server.reports.services import get_reports_db
import cv2
import numpy as np

from config import STATIC_FOLDER
from PySide2.QtMultimedia import QMediaPlayer, QMediaContent
from PySide2.QtMultimediaWidgets import QVideoWidget
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from page_loading_view import LoadingScreen
from PIL import Image


column_ratios = [0.1, 0.15, 0.1, 0.1,0.1,0.15,0.15,0.15]
date_time_format = "yyyy-MM-dd hh:mm:ss"


class PAGEIMAGEVIEW(QDialog):
        def __init__(self, list_image_path=None, video_id=None,path_origin= None):
                super().__init__()

                self.list_path_images = list_image_path
                self.video_id = video_id
                self.path_origin = path_origin

                path_video_dir = f"{STATIC_FOLDER}/videos/{time.strftime('%Y%m%d')}/{os.path.basename(self.path_origin)}"
                if not os.path.exists(path_video_dir):
                        os.makedirs(path_video_dir)
                path_video = f"{path_video_dir}/{self.video_id}.wmv"
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

        def create_video_from_images(self,image_paths, video_name='video_output.avi', output_size=(256, 256), frame_rate=5):
                if len(image_paths) == 0:
                        QMessageBox.warning(self, "Không có hình ảnh", "Không có hình ảnh", QMessageBox.Ok)
                        return
                img = cv2.imread(image_paths[0])
                height, width = img.shape[:2]

                video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), frame_rate, (width, height))

                for index,image_path in enumerate(image_paths):
                        if index < 70:
                                img = cv2.imread(image_path)
                                output_image = np.zeros((height, width, 3), dtype=np.uint8)
                                output_image[0:img.shape[0], 0:img.shape[1], :] = img
                                video.write(output_image)

                cv2.destroyAllWindows()
                video.release()

class PAGEREPORT(QDialog):
        def __init__(self, index, time, analyzer, path_video):
                super().__init__()
                self.list_reports_filter = []
                self.list_file_path = []
                self.counter = 0
                self.video_id = index
                self.time = time
                self.analyzer = analyzer
                self.path_video = path_video
                self.setMinimumSize(QSize(800, 600))
                screen_size = QApplication.primaryScreen().availableSize()
                self.resize(screen_size)
                self.set_ui()
                self.retranslateUi()
                self.setWindowTitle(f"{self.path_video}_{time}")

        def seconds_to_string(self, seconds):
                year, remainder = divmod(seconds, 31536000)
                months, remainder = divmod(seconds, 2592000)
                days, remainder = divmod(seconds, 86400)
                hours, remainder = divmod(remainder, 3600)
                minutes, seconds = divmod(remainder, 60)
                return f"{year:02d}-{months:02d}-{days:02d}-{hours:02d}:{minutes:02d}:{seconds:02d}"
        
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
                self.date_time_start_layout = QHBoxLayout()
                start_label = QLabel("Bắt đầu:", self.filter_groupbox)
                self.date_time_start_layout.addWidget(start_label)
                self.date_time_start_layout.addSpacing(2)
                self.dateTimeEdit_start = QDateTimeEdit(self.filter_groupbox)
                self.dateTimeEdit_start.setObjectName(u"dateTimeEdit_start")
                self.dateTimeEdit_start.setFixedSize(150, 50)
                self.dateTimeEdit_start.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
                start_date = QDateTime.fromString(self.time, date_time_format)
                start_date.setTime(QTime(0, 0, 0))
                self.dateTimeEdit_start.setDateTime(start_date)
                
                self.date_time_start_layout.addWidget(self.dateTimeEdit_start)
                self.date_time_start_layout.addSpacing(20)
                self.date_time_layout.addLayout(self.date_time_start_layout)

                self.date_time_end_layout = QHBoxLayout()
                end_label = QLabel("Kết thúc:", self.filter_groupbox)
                self.date_time_end_layout.addWidget(end_label)
                self.date_time_end_layout.addSpacing(2)
                self.dateTimeEdit_end = QDateTimeEdit(self.filter_groupbox)
                self.dateTimeEdit_end.setObjectName(u"dateTimeEdit_end")
                self.dateTimeEdit_end.setFixedSize(150, 50)
                self.dateTimeEdit_end.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
                end_date = QDateTime.fromString(self.time, date_time_format)
                end_date.setTime(QTime(0, 0, 0))
                self.dateTimeEdit_end.setDateTime(end_date)
                self.date_time_end_layout.addWidget(self.dateTimeEdit_end)
                self.date_time_layout.addLayout(self.date_time_end_layout)

                self.filter_layout.addLayout(self.date_time_layout)
                self.filter_layout.addSpacing(50)

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

                self.checkface_layout = QHBoxLayout()
                checkface_label = QLabel("Nhận diện:", self.filter_groupbox)
                self.checkface_layout.addWidget(checkface_label)
                self.checkface_layout.addSpacing(2)
                self.checkface_combobox = QComboBox(self.filter_groupbox)
                self.checkface_combobox.setObjectName(u"mask_box")
                self.checkface_combobox.setFixedSize(100, 50)
                self.checkface_combobox.addItems(["Tất cả", "Mặt trước", "Mặt sau"])
                self.checkface_layout.addWidget(self.checkface_combobox)
                self.filter_layout.addLayout(self.checkface_layout)
                self.filter_layout.addSpacing(50)

                # Buton Search
                self.search_button = QPushButton("Tìm kiếm", self.filter_groupbox)
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
                self.search_button.clicked.connect(self.filter_report)

                 # Buton Search
                self.import_button = QPushButton("Lọc ảnh", self.filter_groupbox)
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

                # Export report
                self.export_button = QPushButton("Xuất File", self.filter_groupbox)
                self.export_button.setObjectName(u"import_button")
                self.export_button.setFixedSize(100, 50)
                self.filter_layout.addWidget(self.export_button)
                self.export_button.setStyleSheet(u"QPushButton {\n"
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
                icon3.addFile(u":/16x16/icons/16x16/cil-cloud-download.png", QSize(), QIcon.Normal, QIcon.Off)
                self.export_button.setIcon(icon3)
                self.export_button.clicked.connect(self.create_pdf_report)

                # Add the filter group box to the main layout
                self.verticalLayout_6.addWidget(self.filter_groupbox)

                spacer_item = QLabel()
                spacer_item.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
                
                self.filter_layout.addWidget(spacer_item)

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
                        self.tableWidget.cellClicked.connect(self.on_row_selected)

        def retranslateUi(self):
                        font = QFont()
                        font.setFamily("Segoe UI")
                        font.setPointSize(10)
                        font.setBold(True)

                        ___qtablewidgetitem = self.tableWidget.horizontalHeaderItem(0)
                        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"STT", None));
                        ___qtablewidgetitem.setTextColor(QColor(255, 255, 255))
                        ___qtablewidgetitem.setFont(font)
                        ___qtablewidgetitem1 = self.tableWidget.horizontalHeaderItem(1)
                        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"Tên", None));
                        ___qtablewidgetitem1.setTextColor(QColor(255, 255, 255))
                        ___qtablewidgetitem1.setFont(font)
                        ___qtablewidgetitem2 = self.tableWidget.horizontalHeaderItem(2)
                        ___qtablewidgetitem2.setText(QCoreApplication.translate("MainWindow", u"Tuổi", None));
                        ___qtablewidgetitem2.setTextColor(QColor(255, 255, 255))
                        ___qtablewidgetitem2.setFont(font)
                        ___qtablewidgetitem3 = self.tableWidget.horizontalHeaderItem(3)
                        ___qtablewidgetitem3.setText(QCoreApplication.translate("MainWindow", u"Giới tính", None));
                        ___qtablewidgetitem3.setTextColor(QColor(255, 255, 255))
                        ___qtablewidgetitem3.setFont(font)
                        ___qtablewidgetitem4 = self.tableWidget.horizontalHeaderItem(4)
                        ___qtablewidgetitem4.setText(QCoreApplication.translate("MainWindow", u"Đeo khẩu trang", None));
                        ___qtablewidgetitem4.setTextColor(QColor(255, 255, 255))
                        ___qtablewidgetitem4.setFont(font)
                        ___qtablewidgetitem5 = self.tableWidget.horizontalHeaderItem(5)
                        ___qtablewidgetitem5.setText(QCoreApplication.translate("MainWindow", u"Màu Trang Phục", None));
                        ___qtablewidgetitem5.setTextColor(QColor(255, 255, 255))
                        ___qtablewidgetitem5.setFont(font)        
                        ___qtablewidgetitem6 = self.tableWidget.horizontalHeaderItem(6)
                        ___qtablewidgetitem6.setText(QCoreApplication.translate("MainWindow", u"Thời gian nhận diện", None));
                        ___qtablewidgetitem6.setTextColor(QColor(255, 255, 255))
                        ___qtablewidgetitem6.setFont(font)
                        ___qtablewidgetitem7 = self.tableWidget.horizontalHeaderItem(7)
                        ___qtablewidgetitem7.setText(QCoreApplication.translate("MainWindow", u"Ảnh", None));
                        ___qtablewidgetitem7.setTextColor(QColor(255, 255, 255))
                        ___qtablewidgetitem7.setFont(font)


        def get_list_report(self):
                self.list_reports_filter = []
                # Get the Unix timestamp from self.dateTimeEdit_start
                start_timestamp = self.dateTimeEdit_start.dateTime().toSecsSinceEpoch()
                # Get the Unix timestamp from self.dateTimeEdit_end
                end_timestamp = self.dateTimeEdit_end.dateTime().toSecsSinceEpoch()
                if start_timestamp == end_timestamp:
                        start_timestamp = 0
                        end_timestamp = 0
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

                checkface = self.checkface_combobox.currentText()
                if checkface == "Mặt trước":
                        isface = 1
                elif checkface == "Mặt sau":
                        isface = 0
                else:
                        isface = None
                self.list_reports = get_reports_db(self.video_id, page_num, page_size, start_timestamp, end_timestamp, begin_age, end_age, gender, mask, isface)
                print(f"Length list_reports: {len(self.list_reports)}")
                for report in self.list_reports:
                        if len(report['images']) > 0:
                                self.list_reports_filter.append(report)
                        else:
                                print(f"Remove report: {report}")

        def filter_report(self):
                self.get_list_report()
                self.fill_report()
                self.show()

        def convert_timestamp_to_datetime(self,timestamp):
                dt_utc = datetime.utcfromtimestamp(timestamp)
                
                dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                
                dt_vietnam = dt_utc.astimezone(timezone(timedelta(hours=7)))
                
                dt_vietnam_str = dt_vietnam.strftime('%Y-%m-%d %H:%M:%S')
                
                return dt_vietnam_str

        def fill_report(self):
                if len(self.list_reports_filter) >= 16:
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
                                name = "Người không xác định"
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
                        
                        # time_start = QDateTime.fromString(self.time, date_time_format)
                        time_target = QDateTime.fromSecsSinceEpoch(report['time'])
                        time_string = time_target.toString(date_time_format)

                        self.tableWidget.setItem(i, 6, QTableWidgetItem(str(time_string)))

                        list_path_face_image = []
                        list_path_person_image = []
                        # List face image
                        for image_class in report['images']:
                                image_path = image_class['path']
                                if 'face' in os.path.basename(image_path):
                                        list_path_face_image.append(image_path)
                                elif 'person' in os.path.basename(image_path):
                                        list_path_person_image.append(image_path)
                        if len(list_path_face_image) > 0:
                                image_path = list_path_face_image[len(list_path_face_image)//2]
                        else:
                                image_path = list_path_person_image[len(list_path_person_image)//2]
                        print(f"Length face: {len(list_path_face_image)} Length person: {len(list_path_person_image)} path: {image_path}")
                        absolute_image_path = QFileInfo(image_path).absoluteFilePath()
                        os.chmod(image_path, 0o755)
                        image = cv2.imread(absolute_image_path)
                        if image.shape[0] > image.shape[1]:
                                image = cv2.resize(image, (128, int(128 * image.shape[1] / image.shape[0])))
                        else:
                                image = cv2.resize(image, (int(128 * image.shape[0] / image.shape[1]), 128))
                        q_image = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_BGR888)
                        pixmap = QPixmap.fromImage(q_image)
                        item = QTableWidgetItem()
                        item.setData(Qt.DecorationRole, pixmap)
                        self.tableWidget.setItem(i, 7, item)
                        self.tableWidget.setRowHeight(i, pixmap.height())
                        self.tableWidget.setColumnWidth(4, pixmap.width() + 20)

        def on_row_selected(self):
                selected_rows = self.tableWidget.selectionModel().selectedRows()
                list_image_path = []
                if selected_rows:
                        item = [index.row() for index in selected_rows][0]
                        if item >= len(self.list_reports_filter):
                                return
                        list_image = self.list_reports_filter[item]['images']
                        video_id = self.list_reports_filter[item]['id']
                        for image in list_image:
                                path_image = image['path']
                                name_image = os.path.basename(path_image)
                                if "origin_" in name_image:
                                        list_image_path.append(path_image)

                        page_image = PAGEIMAGEVIEW(list_image_path,video_id,self.path_video)
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
                file_dialog.setFileMode(QFileDialog.ExistingFiles)
                file_dialog.setViewMode(QFileDialog.Detail)

                if file_dialog.exec_():
                        self.list_file_path = file_dialog.selectedFiles()
                        if len(self.list_file_path) > 0:
                                loading_screen = LoadingScreen(self)
                                loading_screen.import_loading()
                               
        def create_pdf_report(self):
                try:
                        path_dir = f"{STATIC_FOLDER}\\Documents"
                        if not os.path.exists(path_dir):
                                os.makedirs(path_dir)
                        file_path = f"output_{os.path.basename(self.path_video)}_{time.strftime('%Y%m%d%H%M%S')}.pdf"
                        file_path = os.path.join(path_dir, file_path)
                        font_path = "fonts/segoeui.ttf"
                        pdfmetrics.registerFont(TTFont("Segoe UI", font_path))
                        font_path_b = "fonts/segoeuib.ttf"
                        pdfmetrics.registerFont(TTFont("Segoe UI Bold", font_path_b))
                        # Khởi tạo canvas để vẽ PDF
                        pdf_canvas = canvas.Canvas(file_path, pagesize=letter)

                        items_per_page = 6
                        total_items = len(self.list_reports_filter)
                        total_pages = (total_items + items_per_page - 1) // items_per_page
                        index_report = 0
                        logo_path = "icons/img/photo_2023-12-06_16-22-01.jpg"

                        pdf_canvas.drawInlineImage(logo_path, 250, 710, width=133, height=64)
                        pdf_canvas.setFont("Segoe UI Bold", 20)
                        pdf_canvas.drawString(180, 685, "PHẦN MỀM VIDEOMASTER AI")
                        pdf_canvas.setFont("Segoe UI Bold", 20)
                        pdf_canvas.drawString(215, 660, "BÁO CÁO NHẬN DIỆN")
                        pdf_canvas.setFont("Segoe UI Bold", 11)
                        pdf_canvas.drawString(80, 640, f"Thời gian bắt đầu Video: {self.time}")
                        pdf_canvas.drawString(80, 620, f"Thời gian tạo PDF: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        pdf_canvas.drawString(80, 600, f"Tổng số bản ghi: {total_items}")
                        for page_number in range(total_pages):
                                pdf_canvas.setFont("Segoe UI", 9)
                                # Tạo bảng
                                table_data = [['STT', 'Tên', 'Tuổi', 'Giới Tính', 'Khẩu trang', 'Màu sắc', 'Thời gian', 'Hình ảnh']]

                                if page_number == 0:
                                        end_index = min((page_number + 1) * items_per_page-1, total_items)
                                else:
                                        end_index = min((page_number + 1) * items_per_page, total_items)
                                
                                if page_number == 1:
                                        start_index = page_number * items_per_page -1
                                else:
                                        start_index = page_number * items_per_page

                                for i, report in enumerate(self.list_reports_filter[start_index:end_index]):
                                        index_report += 1
                                        if 'random' in report['person_name']:
                                                name = "Người lạ"
                                        else:
                                                name = report['person_name']
                                        if report['age'] is None:
                                                age = "Không xác định"
                                        else:
                                                age = report['age']
                                        if report['gender'] == 1:
                                                gender = "Nam"
                                        elif report['gender'] == 0:
                                                gender = "Nữ"
                                        else:
                                                gender = "Không xác định"
                                        if report['mask'] == 1:
                                                mask = "Có"
                                        elif report['mask'] == 0:
                                                mask = "Không"
                                        if report['code_color'] is None:
                                                color = None
                                        else:
                                                color = report['code_color']

                                        # time_start = QDateTime.fromString(self.time, date_time_format)
                                        time_target = QDateTime.fromSecsSinceEpoch(report['time'])
                                        date_time_format_m = "yyyy-MM-dd hh:mm"
                                        time_string = time_target.toString(date_time_format_m)

                                        row_data = [
                                                str(index_report),
                                                name,
                                                age,
                                                gender,
                                                mask,
                                                color,
                                                str(time_string),
                                        ]

                                        table_data.append(row_data)

                                # Vẽ bảng
                                row_height = 80
                                col_widths = [40, 40, 60, 60, 60, 60, 60, 65]
                                cell_height = 60
                                cell_widths = [40, 80,60,70,60,65,75,90]

                                for j, header in enumerate(table_data[0]):
                                        pdf_canvas.setFont("Segoe UI Bold", 8, leading=10)
                                        if page_number == 0:
                                                if j == 0:
                                                        x = 30
                                                else:
                                                        x = x_before + cell_widths[j-1]
                                                x_before = x
                                                y = 780 - (0 + 3) * row_height-30
                                                pdf_canvas.rect(x, y, cell_widths[j], cell_height)
                                                # pdf_canvas.rect(j * col_widths[j] + 30, 780- (0 + 3) * row_height, col_widths[j]+25, cell_height)
                                                pdf_canvas.drawString(j * col_widths[j] + 50, 780- (0 + 3) * row_height, header)
                                        else:
                                                if j == 0:
                                                        x = 30
                                                else:
                                                        x = x_before + cell_widths[j-1]
                                                x_before = x
                                                y = 780 - (0 + 1) * row_height-30
                                                pdf_canvas.rect(x, y, cell_widths[j], cell_height)
                                                pdf_canvas.drawString(j * col_widths[j] + 50, 780- (0 + 1) * row_height, header)
                                                # pdf_canvas.rect(j * col_widths[j] + 30, 780- (0 + 1) * row_height, col_widths[j]+25, cell_height)
                                path_color_dir = f"{STATIC_FOLDER}\\Documents\\color"
                                if not os.path.exists(path_color_dir):
                                        os.makedirs(path_color_dir)
                                pdf_canvas.setFont("Segoe UI", 8)
                                for i, row in enumerate(table_data[1:]): 
                                        for j, data in enumerate(row):

                                                pdf_canvas.setFont("Segoe UI", 8, leading=10)
                                                if page_number == 0:
                                                        if j == 0:
                                                                x = 30
                                                        else:
                                                                x = x_before + cell_widths[j-1]
                                                        x_before = x
                                                        y = 780 - (i + 4) * row_height-30
                                                        pdf_canvas.rect(x, y, cell_widths[j], cell_height)
                                                        if j == 5:
                                                                if data is None:
                                                                        pdf_canvas.drawString(j * col_widths[j] + 45, 780 - (i + 4) * row_height, "Không xác định")
                                                                else:
                                                                        data_color = data.split(",")
                                                                        converted_data = [int(element) for element in data_color]
                                                                        image = Image.new("RGB", (20, 10), tuple(converted_data))
                                                                        # Chuyển đổi ảnh thành dữ liệu bytes  
                                                                        
                                                                        image_path = f"{path_color_dir}\\{i}_{data}.png"
                                                                        image.save(image_path)
                                                                        pdf_canvas.drawInlineImage(image_path, j * col_widths[j] + 45, 780 - (i + 4) * row_height-10, width=50, height=30)
                                                        else: 
                                                                pdf_canvas.drawString(j * col_widths[j] + 50, 780 - (i + 4) * row_height, str(data))
                                                                
                                                        # Draw cell border
                                                else:
                                                        if j == 0:
                                                                x = 30
                                                        else:
                                                                x = x_before + cell_widths[j-1]
                                                        x_before = x
                                                        y = 780 - (i + 2) * row_height-30
                                                        pdf_canvas.rect(x, y, cell_widths[j], cell_height)
                                                        if j == 5:
                                                                if data is None:
                                                                        pdf_canvas.drawString(j * col_widths[j] + 45, 780 - (i + 2) * row_height, "Không xác định")
                                                                else:
                                                                        data_color = data.split(",")
                                                                        converted_data = [int(element) for element in data_color]
                                                                        image = Image.new("RGB", (20, 10), tuple(converted_data))
                                                                        # Chuyển đổi ảnh thành dữ liệu bytes  
                                                                        image_path = f"{path_color_dir}\\{i}_{data}.png"
                                                                        image.save(image_path)
                                                                        pdf_canvas.drawInlineImage(image_path, j * col_widths[j] + 45, 780 - (i + 2) * row_height-10, width=50, height=30)
                                                        else:
                                                                if data == "Không xác định":
                                                                        pdf_canvas.drawString(j * col_widths[j] + 45, 780 - (i + 2) * row_height, str(data))
                                                                else:
                                                                        pdf_canvas.drawString(j * col_widths[j] + 50, 780 - (i + 2) * row_height, str(data))

                                for i, report in enumerate(self.list_reports_filter[start_index:end_index]):
                                        
                                        if len(report.get('images', [])) > 0:
                                                image_path = report['images'][0]['path']
                                                list_path_face_image = []
                                                list_path_person_image = []
                                                # List face image
                                                for image_class in report['images']:
                                                        image_path = image_class['path']
                                                        if 'face' in os.path.basename(image_path):
                                                                list_path_face_image.append(image_path)
                                                        elif 'person' in os.path.basename(image_path):
                                                                list_path_person_image.append(image_path)
                                                if len(list_path_face_image) > 0:
                                                        image_path = list_path_face_image[len(list_path_face_image)//2]
                                                else:
                                                        image_path = list_path_person_image[len(list_path_person_image)//2]
                                                if page_number==0:
                                                        x = x_before + 75
                                                        y = 780 - (i + 4) * row_height-30
                                                        pdf_canvas.rect(x, y, 90, 60)
                                                        pdf_canvas.drawInlineImage(image_path, 50 + 440, 780 - (i+4) * row_height-30, width=60, height=60)
                                                else:
                                                        x = x_before + 75
                                                        y = 780 - (i + 2) * row_height-30
                                                        pdf_canvas.rect(x, y, 90, 60)
                                                        pdf_canvas.drawInlineImage(image_path, 50 + 440, 780 - (i+2) * row_height-30, width=60, height=60)
                                        else:
                                                if page_number==0:
                                                        x = x_before + 75
                                                        y = 780 - (i + 4) * row_height-30
                                                        pdf_canvas.rect(x, y, 90, 60)
                                                        pdf_canvas.drawString(50 + 440, 780 - (i+4) * row_height-20, "Không có hình ảnh")
                                                else:
                                                        x = x_before + 75
                                                        y = 780 - (i + 2) * row_height-30
                                                        pdf_canvas.rect(x, y, 90, 60)
                                                        pdf_canvas.drawString(50 + 440, 780 - (i+2) * row_height-20, "Không có hình ảnh")
                                
                                pdf_canvas.showPage()
                        # Lưu file PDF
                        pdf_canvas.save()
                        QMessageBox.information(self, "Đã lưu", "File pdf đã được lưu thành công", QMessageBox.Ok)
                        if os.path.exists(file_path):
                        # Sử dụng subprocess để mở file
                                try:
                                        subprocess.Popen(['start', '', file_path], shell=True)
                                except Exception as e:
                                        print(f"Không thể mở file: {e}")
                        else:
                                print("File không tồn tại")
                except Exception as e:
                        print(f"[save_pdf]: {e}")
                        QMessageBox.warning(self, "Không thể lưu", "Không thể lưu file pdf", QMessageBox.Ok)

                                                        
