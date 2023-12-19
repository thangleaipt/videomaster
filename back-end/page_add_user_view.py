
from PySide2.QtCore import (QCoreApplication, QMetaObject, QObject, QPoint,
    QRect, QSize, QUrl, Qt)
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter,QPixmap,QImage,
    QRadialGradient)

from PyQt5.QtCore import QTimer
from PySide2.QtWidgets import *

import cv2

class PAGEADDUSER(QWidget):
    def __init__(self,page_video, page_image):
        super().__init__()
        self.selected_labels = []
        self.selected_labels_2 = []
        self.upload_image = None
        self.analyzer = None
        self.page_video = page_video
        self.page_image = page_image
        self.setObjectName(u"page_user")
        
    def set_ui(self):
        self.main_layout = QVBoxLayout(self)
        self.add_layout = QHBoxLayout()
        self.camera_layout = QVBoxLayout()

        # Image
        self.camera_labels = []
        camera_label = QLabel(self)
        camera_label.setObjectName(u"camera_add_user")
        camera_label.setStyleSheet("border: 2px solid red;")
        # Set size of camera according to screen
        camera_label.setMinimumSize(QSize(720, 600))
        camera_label.setMaximumSize(QSize(16777215, 16777215))
        #center
        camera_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        camera_label.setAlignment(Qt.AlignCenter)
        self.camera_labels.append(camera_label)
        self.camera_layout.addWidget(camera_label)
        # Button upload image
        self.uploadButton = QPushButton(self)
        self.uploadButton.setObjectName(u"uploadButton")
        font8 = QFont()
        font8.setFamily(u"Segoe UI")
        font8.setPointSize(9)
        self.uploadButton.setFont(font8)
        self.uploadButton.setStyleSheet(u"QPushButton {\n"
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
        icon3.addFile(u":/16x16/icons/16x16/cil-folder-open.png", QSize(), QIcon.Normal, QIcon.Off)
        self.uploadButton.setIcon(icon3)
        self.uploadButton.setText(QCoreApplication.translate("MainWindow", u"Open Image", None))
        self.uploadButton.setMinimumSize(QSize(200, 30))
        self.uploadButton.setMaximumSize(QSize(200, 30))
        self.uploadButton.clicked.connect(self.open_video_file)
        self.camera_layout.addWidget(self.uploadButton)
        self.add_layout.addLayout(self.camera_layout)

        self.controls_layout = QVBoxLayout()
        self.lineLabel = QLineEdit(self)
        self.lineLabel.setObjectName(u"lineEdit")
        self.lineLabel.setMinimumSize(QSize(0, 30))
        self.lineLabel.setMaximumSize(QSize(600, 30))
        self.lineLabel.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Họ và tên", None))
        self.lineLabel.setStyleSheet(u"QLineEdit {\n"
            "	background-color: rgb(27, 29, 35);\n"
            "	border-radius: 5px;\n"
            "	border: 2px solid rgb(27, 29, 35);\n"
            "	padding-left: 10px;\n"
            "}\n"
            "QLineEdit:hover {\n"
            "	border: 2px solid rgb(64, 71, 88);\n"
            "}\n"
            "QLineEdit:focus {\n"
            "	border: 2px solid rgb(91, 101, 124);\n"
            "}")
        self.controls_layout.addWidget(self.lineLabel)

        # Position
        self.linePosition = QLineEdit(self)
        self.linePosition.setObjectName(u"lineEdit")
        self.linePosition.setMinimumSize(QSize(0, 30))
        self.linePosition.setMaximumSize(QSize(600, 30))
        self.linePosition.setPlaceholderText(QCoreApplication.translate("MainWindow", u"Đối tượng", None))
        self.linePosition.setStyleSheet(u"QLineEdit {\n"
            "	background-color: rgb(27, 29, 35);\n"
            "	border-radius: 5px;\n"
            "	border: 2px solid rgb(27, 29, 35);\n"
            "	padding-left: 10px;\n"
            "}\n"
            "QLineEdit:hover {\n"
            "	border: 2px solid rgb(64, 71, 88);\n"
            "}\n"
            "QLineEdit:focus {\n"
            "	border: 2px solid rgb(91, 101, 124);\n"
            "}")
        self.controls_layout.addWidget(self.linePosition)

        # Button add user
        self.addButton = QPushButton(self)
        self.addButton.setObjectName(u"addButton")
        font8 = QFont()
        font8.setFamily(u"Segoe UI")
        font8.setPointSize(9)
        self.addButton.setFont(font8)
        self.addButton.setStyleSheet(u"QPushButton {\n"
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
        icon3.addFile(u":/16x16/icons/16x16/cil-user-follow.png", QSize(), QIcon.Normal, QIcon.Off)
        self.addButton.setIcon(icon3)
        self.addButton.setText(QCoreApplication.translate("MainWindow", u"Add User", None))
        self.addButton.setMinimumSize(QSize(200, 30))
        self.addButton.setMaximumSize(QSize(200, 30))
        self.addButton.clicked.connect(self.add_label_images)
        self.controls_layout.addWidget(self.addButton)

        # Button delete user
        self.deleteButton = QPushButton(self)
        self.deleteButton.setObjectName(u"deleteButton")
        self.deleteButton.setEnabled(False)
        font8 = QFont()
        font8.setFamily(u"Segoe UI")
        font8.setPointSize(9)
        self.deleteButton.setFont(font8)
        self.deleteButton.setStyleSheet(u"QPushButton {\n"
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
        icon3.addFile(u":/16x16/icons/16x16/cil-user-unfollow.png", QSize(), QIcon.Normal, QIcon.Off)
        self.deleteButton.setIcon(icon3)
        self.deleteButton.setText(QCoreApplication.translate("MainWindow", u"Delete User", None))
        self.deleteButton.setMinimumSize(QSize(200, 30))
        self.deleteButton.setMaximumSize(QSize(200, 30))
        self.deleteButton.clicked.connect(self.delete_selected_labels)
        self.controls_layout.addWidget(self.deleteButton)

        self.add_layout.addLayout(self.controls_layout)
       
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_content = QWidget()
        self.ref_layout = QHBoxLayout(scroll_content)
        self.update_ui()

        # Set the content widget for the scroll area
        scroll_area.setWidget(scroll_content)
        # Set size of scroll area according to screen size
        scroll_area.setMinimumSize(QSize(16777215, 300))
        scroll_area.setMaximumSize(QSize(16777215, 350))

        # Thêm grid_layout và QScrollArea vào main_layout
        self.main_layout.addLayout(self.add_layout)
        self.main_layout.addWidget(scroll_area)

    def update_ui(self):
        # Clear the existing layout
        self.clear_layout(self.ref_layout)

        # Create and add QLabel widgets to the layout
        for i, employee in enumerate(self.analyzer.representations):
            label_layout = QVBoxLayout()

            # Employee Label
            employee_label = QLabel(f"{employee[2]}")
            employee_label.setAlignment(Qt.AlignCenter)
            label_layout.addWidget(employee_label)

            # Add spacing between employee_label and image_label
            label_layout.addSpacing(10)  # Adjust the spacing as needed

            # Recognition Image
            recognition_image = cv2.imread(employee[0])
            recognition_image = cv2.cvtColor(recognition_image, cv2.COLOR_BGR2RGB)
            recognition_image = cv2.resize(recognition_image, (200, recognition_image.shape[0] * 200 // recognition_image.shape[1]))
            recognize_image = QImage(recognition_image.data, recognition_image.shape[1], recognition_image.shape[0], QImage.Format_RGB888)
            image_label = QLabel()
            image_label.setPixmap(QPixmap.fromImage(recognize_image))
            image_label.setAlignment(Qt.AlignCenter)
            image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            label_layout.addWidget(image_label)
            self.ref_layout.addLayout(label_layout)
            image_label.mousePressEvent = lambda event, l=(employee_label, image_label): self.select_label(l)

    def clear_layout(self, layout):
        if layout is None:
            return
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
            else:
                self.clear_layout(item.layout())

    def open_video_file(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Image files ( *.png *.jpg *.jpeg *.bmp)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setViewMode(QFileDialog.Detail)

        if file_dialog.exec_():
                file_path = file_dialog.selectedFiles()
                if file_path:
                    print("Selected file:", file_path[0])
                    self.upload_image = cv2.imread(file_path[0])
                    image = cv2.cvtColor(self.upload_image, cv2.COLOR_BGR2RGB)
                    height, width, channel = image.shape
                    step = channel * width
                    # create QImage from image
                    qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
                    # show image in img_label
                    for camera in self.camera_labels:
                        camera.setPixmap(QPixmap.fromImage(qImg))

    def select_label(self, labels):
        if labels[0] in self.selected_labels:
            self.selected_labels.remove(labels[0])
            labels[1].setStyleSheet("")  
            self.selected_labels_2.remove(labels[1])
        else:
            self.selected_labels.append(labels[0])
            self.selected_labels_2.append(labels[1])
            labels[1].setStyleSheet("border: 2px solid red;")  

        self.deleteButton.setEnabled(bool(self.selected_labels))

    def delete_selected_labels(self):
        for i, label in enumerate(self.selected_labels):
            label.deleteLater()  
            self.selected_labels[i].deleteLater()
            self.analyzer.delete(self.selected_labels[i].text())
            for key, value in self.page_video.list_camera_screen.items():
                value.worker.face_analyzer.load_db_from_folder()
            for key, value in self.page_image.list_camera_screen.items():
                value.worker.face_analyzer.load_db_from_folder()
        self.selected_labels = [] 
        self.selected_labels_2 = [] 
        self.deleteButton.setEnabled(False)
        self.update_ui()
        QMessageBox.information(self, "Hoàn thành", f"Đã xóa!")

    def add_label_images(self):
        try:
            cur_label = self.lineLabel.text()
            cur_position = self.linePosition.text()
            img = self.upload_image
            if not cur_label:
                print("empty label")
                return
            # for i in range(len(self.selected_labels)):
            self.analyzer.train(img,cur_label,cur_position)
            for key, value in self.page_video.list_camera_screen.items():
                value.worker.face_analyzer.load_db_from_folder()
            # reset image
            self.camera_labels[0].setPixmap(QPixmap())
            self.lineLabel.setText("")
            self.linePosition.setText("")
            self.update_ui()
            QMessageBox.information(self, "Hoàn thành", f"{cur_label} đã được thêm vào hệ thống!")

        except Exception as e:
            print(f"[add_label_images]: {e}")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.newsize = event.size()
        self.uploadButton.setFixedSize(int(self.newsize.width()/10* 2), int(self.newsize.height()/20))
        self.deleteButton.setFixedSize(int(self.newsize.width()/10* 2), int(self.newsize.height()/20))
        self.addButton.setFixedSize(int(self.newsize.width()/10* 2), int(self.newsize.height()/20))
        self.lineLabel.setFixedSize(int(self.newsize.width()/10* 2), int(self.newsize.height()/20))
        self.linePosition.setFixedSize(int(self.newsize.width()/10* 2), int(self.newsize.height()/20))

                              

