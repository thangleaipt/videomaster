import sys
from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QDateTimeEdit, QPushButton

class DateTimePickerApp(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()

    def init_ui(self):
        # Tạo ô nhập ngày giờ
        self.datetime_edit = QDateTimeEdit(self)
        self.datetime_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")

        # Tạo nút để in ra ngày giờ được chọn
        btn_get_datetime = QPushButton("Get Date and Time", self)
        btn_get_datetime.clicked.connect(self.get_selected_datetime)

        # Sắp xếp layout
        layout = QVBoxLayout(self)
        layout.addWidget(self.datetime_edit)
        layout.addWidget(btn_get_datetime)

        self.setLayout(layout)

        self.setWindowTitle('Date and Time Picker')
        self.setGeometry(300, 300, 400, 150)

    def get_selected_datetime(self):
        selected_datetime = self.datetime_edit.dateTime()
        print("Selected date and time:", selected_datetime.toString("yyyy-MM-dd HH:mm:ss"))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DateTimePickerApp()
    window.show()
    sys.exit(app.exec_())
