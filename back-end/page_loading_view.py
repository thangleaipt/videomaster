import sys
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *
from ui_loading_screen import Ui_LoadingScreen
from widgets_loading import CircularProgress

class WorkerSignals(QObject):
    finished = Signal()

class Worker(QRunnable):
    def __init__(self, signals):
        super(Worker, self).__init__()
        self.signals = signals

    @Slot()
    def run(self):
        for i in range(101):
            self.signals.finished.emit()
            QThread.msleep(25)  # Simulate some work
        self.signals.finished.emit()

class LoadingScreen(QMainWindow):
    def __init__(self, threadpool):
        super(LoadingScreen, self).__init__()
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

        self.worker_signals = WorkerSignals()
        self.worker = Worker(self.worker_signals)
        self.worker.signals.finished.connect(self.close_loading_screen)

        self.threadpool = threadpool

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_status)
        self.timer.start(25)

        self.show()

    def update_status(self):
        global counter  # Add this line to make counter global
        self.progress.set_value(counter)
        if counter >= 500:
            self.timer.stop()
            self.threadpool.start(self.worker)
        counter += 1

    def close_loading_screen(self):
        self.close()

def main():
    global counter  # Add this line to make counter global
    counter = 0

    app = QApplication(sys.argv)
    threadpool = QThreadPool.globalInstance()

    loading_screen = LoadingScreen(threadpool)

    # Continue with the rest of your application logic here

    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
