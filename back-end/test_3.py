import sys
from PySide2.QtWidgets import QApplication, QMainWindow, QPushButton


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Main Window")
        self.setGeometry(100, 100, 400, 200)

        button = QPushButton("Open Second Window", self)
        button.clicked.connect(self.open_second_window)

    def open_second_window(self):
        second_window = SecondMainWindow()
        second_window.show()


class SecondMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Second Main Window")
        self.setGeometry(500, 100, 400, 200)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
