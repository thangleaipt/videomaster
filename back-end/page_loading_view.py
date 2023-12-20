
import time
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from ui_loading_screen import Ui_LoadingScreen
from widgets_loading import CircularProgress
from PIL import Image

column_ratios = [0.1, 0.15, 0.1, 0.1,0.1,0.15,0.15,0.15]

class FilterThread(QThread):
    finished = Signal()
    progress_update = Signal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report = parent

    def run(self):
        self.report.get_list_report()
        self.fill_report()
        self.finished.emit()
    
    def fill_report(self):
                if len(self.report.list_reports_filter) >= 16:
                        self.report.tableWidget.setRowCount(len(self.report.list_reports_filter))
                else:
                        if len(self.report.list_reports_filter) == 0:
                                QMessageBox.information(self.report, "Notification", "Không tìm thấy kết quả.")
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
                                name = "Người lạ"
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
                        
                        self.report.tableWidget.setItem(i, 6, QTableWidgetItem(str(self.report.convert_timestamp_to_datetime(report['time']))))
                        if len(report['images']) > 0:
                                image_path = report['images'][0]['path']
                                pixmap = QPixmap(image_path).scaledToWidth(128, Qt.SmoothTransformation).scaledToHeight(128, Qt.SmoothTransformation)
                                item = QTableWidgetItem()
                                item.setData(Qt.DecorationRole, pixmap)
                                self.report.tableWidget.setItem(i, 7, item)

                                self.report.tableWidget.setRowHeight(i, pixmap.height())

                                self.report.tableWidget.setColumnWidth(4, pixmap.width() + 20)

                self.report.tableWidget.cellClicked.connect(self.report.on_row_selected)

class ImportThread(QThread):
    finished = Signal()
    progress_update = Signal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self.report = parent

    def run(self):
        self.report.filter_report_query()
        self.fill_report()
        self.finished.emit()

    def fill_report(self):
                if len(self.report.list_reports_filter) >= 16:
                        self.report.tableWidget.setRowCount(len(self.report.list_reports_filter))
                else:
                        if len(self.report.list_reports_filter) == 0:
                                QMessageBox.information(self.report, "Notification", "Không tìm thấy kết quả.")
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
                                name = "Người lạ"
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
                        
                        self.report.tableWidget.setItem(i, 6, QTableWidgetItem(str(self.report.convert_timestamp_to_datetime(report['time']))))
                        if len(report['images']) > 0:
                                print(f"report['images']: Initial:")
                                image_path = report['images'][0]['path']

                                pixmap = QPixmap(image_path).scaledToWidth(128, Qt.SmoothTransformation).scaledToHeight(128, Qt.SmoothTransformation)
                                item = QTableWidgetItem()
                                item.setData(Qt.DecorationRole, pixmap)
                                self.report.tableWidget.setItem(i, 7, item)

                                self.report.tableWidget.setRowHeight(i, pixmap.height())

                                self.report.tableWidget.setColumnWidth(4, pixmap.width() + 20)

                self.report.tableWidget.cellClicked.connect(self.report.on_row_selected)

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

