
from PySide2.QtCore import *
from PySide2.QtGui import *
from PySide2.QtWidgets import *


class Ui_LoadingScreen(object):
    def setupUi(self, LoadingScreen):
        if not LoadingScreen.objectName():
            LoadingScreen.setObjectName(u"LoadingScreen")
        LoadingScreen.resize(350, 350)
        LoadingScreen.setMinimumSize(QSize(350, 350))
        LoadingScreen.setMaximumSize(QSize(350, 350))
        self.centralwidget = QWidget(LoadingScreen)
        self.centralwidget.setObjectName(u"centralwidget")
        self.verticalLayout = QVBoxLayout(self.centralwidget)
        self.verticalLayout.setSpacing(0)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.verticalLayout.setContentsMargins(10, 10, 10, 10)
        self.container = QFrame(self.centralwidget)
        self.container.setObjectName(u"container")
        self.container.setFrameShape(QFrame.NoFrame)
        self.container.setFrameShadow(QFrame.Raised)
        self.verticalLayout_2 = QVBoxLayout(self.container)
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.verticalLayout_2.setContentsMargins(20, 20, 20, 20)
        self.background = QFrame(self.container)
        self.background.setObjectName(u"background")
        self.background.setEnabled(True)
        self.background.setStyleSheet(u"QFrame{\n"
"	background-color: #221d23;\n"
"	color: #DDB967;\n"
"	border-radius: 145px;\n"
"	font: 63 8pt \"Yu Gothic UI Semibold\";\n"
"\n"
"}\n"
"")
        self.background.setFrameShape(QFrame.NoFrame)
        self.background.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.background)
        self.verticalLayout_3.setSpacing(0)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.message = QFrame(self.background)
        self.message.setObjectName(u"message")
        self.message.setMaximumSize(QSize(16777215, 160))
        self.message.setStyleSheet(u"background: none;")
        self.message.setFrameShape(QFrame.NoFrame)
        self.message.setFrameShadow(QFrame.Raised)
        self.verticalLayout_4 = QVBoxLayout(self.message)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.gridLayout = QGridLayout()
        self.gridLayout.setObjectName(u"gridLayout")
        self.title = QLabel(self.message)
        self.title.setObjectName(u"title")
        self.title.setMinimumSize(QSize(0, 24))
        self.title.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.title, 0, 0, 1, 1)

        self.loading = QLabel(self.message)
        self.loading.setObjectName(u"loading")
        self.loading.setMinimumSize(QSize(0, 20))
        self.loading.setAlignment(Qt.AlignCenter)

        self.gridLayout.addWidget(self.loading, 3, 0, 1, 1)

        self.frame = QFrame(self.message)
        self.frame.setObjectName(u"frame")
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.frame)
        self.verticalLayout_5.setSpacing(0)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(0, 0, 0, 0)
        self.example = QLabel(self.frame)
        self.example.setObjectName(u"example")
        self.example.setMinimumSize(QSize(100, 20))
        self.example.setMaximumSize(QSize(100, 22))
        self.example.setStyleSheet(u"QLabel{\n"
"	background-color: #4e4250;\n"
"	color: #D0E37F;\n"
"	border-radius: 10px;\n"
"}")
        self.example.setAlignment(Qt.AlignCenter)

        self.verticalLayout_5.addWidget(self.example, 0, Qt.AlignHCenter)


        self.gridLayout.addWidget(self.frame, 2, 0, 1, 1)

        self.frame_2 = QFrame(self.message)
        self.frame_2.setObjectName(u"frame_2")
        self.frame_2.setMinimumSize(QSize(270, 80))
        self.frame_2.setMaximumSize(QSize(270, 80))
        self.frame_2.setFrameShape(QFrame.NoFrame)
        self.frame_2.setFrameShadow(QFrame.Raised)

        self.gridLayout.addWidget(self.frame_2, 1, 0, 1, 1)


        self.verticalLayout_4.addLayout(self.gridLayout)


        self.verticalLayout_3.addWidget(self.message)


        self.verticalLayout_2.addWidget(self.background)


        self.verticalLayout.addWidget(self.container)

        LoadingScreen.setCentralWidget(self.centralwidget)

        self.retranslateUi(LoadingScreen)

        QMetaObject.connectSlotsByName(LoadingScreen)
    # setupUi

    def retranslateUi(self, LoadingScreen):
        LoadingScreen.setWindowTitle(QCoreApplication.translate("LoadingScreen", u"LoadingWindow", None))
        self.title.setText(QCoreApplication.translate("LoadingScreen", u"Simple loading screen", None))
        self.loading.setText(QCoreApplication.translate("LoadingScreen", u"Loading___", None))
        self.example.setText(QCoreApplication.translate("LoadingScreen", u"Example", None))
    # retranslateUi

