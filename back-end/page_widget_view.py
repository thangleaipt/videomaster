from datetime import datetime, timezone, timedelta

from PySide2.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt, QDateTime, QTime)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter,QPixmap,QImage,
    QRadialGradient)

from PySide2.QtWidgets import *
from page_report_view import PAGEREPORT
from server.reports.services import get_videos_path_db


column_ratios = [0.1, 0.1, 0.3, 0.3, 0.2]

class PAGEWIDGET(QWidget):
        def __init__(self):
                super().__init__()
                self.analyzer = None
                self.setObjectName(u"page_widget")
                self.set_ui()
                self.retranslateUi()
                self.setWindowTitle("Báo cáo dữ liệu")

        def set_ui(self):
                self.verticalLayout_6 = QVBoxLayout(self)
                self.verticalLayout_6.setObjectName(u"verticalLayout_6")

                # Create a group box for the filter controls
                font = QFont("Segoe UI", 15)
                font.setBold(True)
                self.filter_groupbox = QGroupBox("Lọc dữ liệu", self)
                self.filter_groupbox.setFont(font)
                self.filter_groupbox.setObjectName(u"filter_groupbox")

                # Create a layout for the filter group box
                self.filter_layout = QHBoxLayout(self.filter_groupbox)
                self.filter_layout.setObjectName(u"filter_layout")

                self.date_time_layout = QHBoxLayout(self)

                # Add date-time edit controls to the filter layout
                start_label = QLabel("Start Date:", self.filter_groupbox)
                self.date_time_layout.addWidget(start_label)
                self.date_time_layout.addSpacing(2)
                self.dateTimeEdit_start = QDateTimeEdit(self.filter_groupbox)
                self.dateTimeEdit_start.setObjectName(u"dateTimeEdit_start")
                self.dateTimeEdit_start.setFixedSize(200, 50)
                self.dateTimeEdit_start.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
                start_date = QDateTime.currentDateTime()
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
                end_date = QDateTime.currentDateTime()
                end_date.setTime(QTime(23, 59, 59))
                self.dateTimeEdit_end.setDateTime(end_date)
                self.date_time_layout.addWidget(self.dateTimeEdit_end)

                self.filter_layout.addLayout(self.date_time_layout)
                self.filter_layout.addSpacing(50)

                # Buton Search
                self.search_button = QPushButton("Search", self.filter_groupbox)
                self.search_button.setObjectName(u"search_button")
                self.search_button.setFixedSize(100, 50)
                self.filter_layout.addWidget(self.search_button)
                self.search_button.setStyleSheet(u"QPushButton {\n"
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
                icon3.addFile(u":/16x16/icons/16x16/cil-magnifying-glass.png", QSize(), QIcon.Normal, QIcon.Off)
                self.search_button.setIcon(icon3)
                self.search_button.clicked.connect(self.get_list_path_video)

                # Add the filter group box to the main layout
                self.verticalLayout_6.addWidget(self.filter_groupbox)


                self.frame_3 = QFrame(self)
                self.frame_3.setObjectName(u"frame_3")
                self.frame_3.setMinimumSize(QSize(0, 150))
                self.frame_3.setFrameShape(QFrame.StyledPanel)
                self.frame_3.setFrameShadow(QFrame.Raised)
                self.horizontalLayout_12 = QHBoxLayout(self.frame_3)
                self.horizontalLayout_12.setSpacing(0)
                self.horizontalLayout_12.setObjectName(u"horizontalLayout_12")
                self.horizontalLayout_12.setContentsMargins(0, 0, 0, 0)
                self.tableWidget = QTableWidget(self.frame_3)
                if (self.tableWidget.columnCount() < 5):
                        self.tableWidget.setColumnCount(5)
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
                        ___qtablewidgetitem = self.tableWidget.horizontalHeaderItem(0)
                        ___qtablewidgetitem.setText(QCoreApplication.translate("MainWindow", u"STT", None));
                        ___qtablewidgetitem1 = self.tableWidget.horizontalHeaderItem(1)
                        ___qtablewidgetitem1.setText(QCoreApplication.translate("MainWindow", u"ID", None));
                        ___qtablewidgetitem2 = self.tableWidget.horizontalHeaderItem(2)
                        ___qtablewidgetitem2.setText(QCoreApplication.translate("MainWindow", u"Path Video", None));
                        ___qtablewidgetitem3 = self.tableWidget.horizontalHeaderItem(3)
                        ___qtablewidgetitem3.setText(QCoreApplication.translate("MainWindow", u"Thời gian bắt đầu", None));
                        ___qtablewidgetitem4 = self.tableWidget.horizontalHeaderItem(4)
                        ___qtablewidgetitem4.setText(QCoreApplication.translate("MainWindow", u"Thời gian upload", None));
                        self.get_list_path_video()

        def convert_timestamp_to_datetime(self,timestamp):
                dt_utc = datetime.utcfromtimestamp(timestamp)
                
                dt_utc = dt_utc.replace(tzinfo=timezone.utc)
                
                dt_vietnam = dt_utc.astimezone(timezone(timedelta(hours=7)))
                
                dt_vietnam_str = dt_vietnam.strftime('%Y-%m-%d %H:%M:%S')
                
                return dt_vietnam_str
        def get_list_path_video(self):
                # Get the Unix timestamp from self.dateTimeEdit_start
                start_timestamp = self.dateTimeEdit_start.dateTime().toSecsSinceEpoch()
                # Get the Unix timestamp from self.dateTimeEdit_end
                end_timestamp = self.dateTimeEdit_end.dateTime().toSecsSinceEpoch()
                page_num = None
                page_size = None

                list_path_video = get_videos_path_db(page_num=page_num, page_size=page_size, start_time=start_timestamp, end_time=end_timestamp)
                if len(list_path_video) >= 16:
                        self.tableWidget.setRowCount(len(list_path_video))
                for i, path_video in enumerate(list_path_video):
                        self.tableWidget.setItem(i, 0, QTableWidgetItem(str(i)))
                        self.tableWidget.setItem(i, 1, QTableWidgetItem(str(path_video.id)))
                        self.tableWidget.setItem(i, 2, QTableWidgetItem(str(path_video.path)))
                        self.tableWidget.setItem(i, 3, QTableWidgetItem(str(self.convert_timestamp_to_datetime(path_video.start_time))))
                        self.tableWidget.setItem(i, 4, QTableWidgetItem(str(self.convert_timestamp_to_datetime(path_video.time))))
                return list_path_video
        
        def on_row_selected(self):
                selected_rows = self.tableWidget.selectionModel().selectedRows()
                if selected_rows:
                        for index in selected_rows:
                                item = self.tableWidget.item(index.row(), 1)
                                item_time = self.tableWidget.item(index.row(), 3)
                                if item is None or item_time is None:
                                        continue
                                video_id = int(item.text())
                                # Fomat datetime
                                time = (item_time.text())
                                path_video = self.tableWidget.item(index.row(), 2).text()
                                page_report = PAGEREPORT(video_id, time, self.analyzer, path_video)
                                page_report.show()
          
        def resizeEvent(self,event):
                screen_width = event.size().width()
                column_widths = [int(ratio * screen_width) for ratio in column_ratios]
                for i in range(5):
                        self.tableWidget.setColumnWidth(i, column_widths[i])
                super().resizeEvent(event)
