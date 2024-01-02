from PySide2.QtWidgets import QDialog, QVBoxLayout, QPushButton, QApplication
from PySide2.QtMultimedia import QMediaPlayer, QMediaContent
from PySide2.QtCore import QUrl
from PySide2.QtMultimediaWidgets import QVideoWidget

class VideoDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Video Dialog")
        self.setGeometry(100, 100, 800, 600)

        # Video widget
        self.video_widget = QVideoWidget(self)

        # Media player
        self.media_player = QMediaPlayer(self)
        self.media_player.setVideoOutput(self.video_widget)

        # Play button
        play_button = QPushButton("Play", self)
        play_button.clicked.connect(self.play_video)

        # Stop button
        stop_button = QPushButton("Stop", self)
        stop_button.clicked.connect(self.stop_video)

        # Layout setup
        layout = QVBoxLayout()
        layout.addWidget(play_button)
        layout.addWidget(stop_button)
        layout.addWidget(self.video_widget)

        self.setLayout(layout)

        # Load a sample video (replace with your own path)
        video_path = r"D:\PROJEC_THANGLT\HumanMaster_app\back-end\static\videos\20231207\16085.mp4"
        video_url = QUrl.fromLocalFile(video_path)
        content = QMediaContent(video_url)
        self.media_player.setMedia(content)

    def play_video(self):
        self.media_player.play()

    def stop_video(self):
        self.media_player.stop()

if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    dialog = VideoDialog()
    dialog.exec_()

    sys.exit(app.exec_())
