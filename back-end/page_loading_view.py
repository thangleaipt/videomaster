import sys

from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *

from ui_loading_screen import Ui_LoadingScreen
from widgets_loading import CircularProgress

import sys
from PySide2.QtCore import Qt, QThread, Signal
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton
from ui_loading_screen import Ui_LoadingScreen

class FilterThread(QThread):
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.report = parent

    def run(self):
        self.report.get_list_report()
        self.report.fill_report()
        self.finished.emit()

class ImportThread(QThread):
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.report = parent

    def run(self):
        self.report.filter_report_query()
        self.report.fill_report()
        self.finished.emit()

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
        self.loading_thread.start()
    
    def import_loading(self):
        self.loading_thread = ImportThread(self.report)
        self.loading_thread.finished.connect(self.loading_finished)
        self.loading_thread.start()

    def loading_finished(self):
        self.close()
        self.report.show()

