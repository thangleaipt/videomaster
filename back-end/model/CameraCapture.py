import queue, threading
from PySide2.QtGui import (QBrush, QColor, QConicalGradient, QCursor, QFont,
    QFontDatabase, QIcon, QLinearGradient, QPalette, QPainter,QPixmap,QImage,
    QRadialGradient)
from controller.Face_recognition.analyze_video_insightface import FaceAnalysisInsightFace

# bufferless VideoCapture
class CameraCapture:
    def __init__(self, capdevice):
        self.cap = capdevice
        self.q = queue.Queue(1)
        self.no_stop_request = True
        self.t = threading.Thread(target=self._reader)
        self.t.daemon = True
        self.t.start()

    # read frames as soon as they are available, keeping only most recent one
    def _reader(self):
        while self.no_stop_request:
            ret, frame = self.cap.read()

            if not self.q.empty():
                try:
                    self.q.get_nowait()  # discard previous(unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        frame = self.q.get()
        if frame is not None:
            ret = True
        else:
            ret = False
        return ret, frame

    def close(self, timeout_sec):
        self.no_stop_request = False
        if self.t.is_alive():
            self.t.join(timeout_sec)

from threading import Thread
import cv2
import time

class VideoWriterWidget():
    def __init__(self, src=0,camera_labels=[]):
        self.camera_labels = camera_labels
        # Create a VideoCapture object
        self.frame_name = str(src) # if using webcams, else just use src as it is.
        self.video_file = f"{src}_output"
        self.video_file_name = f"{src}_output" + '.avi'
        self.capture = cv2.VideoCapture(src)
        self.face_analyzer = FaceAnalysisInsightFace()

        self.index_frame = 0
        # Default resolutions of the frame are obtained (system dependent)
        self.frame_width = int(self.capture.get(3))
        self.frame_height = int(self.capture.get(4))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.count = self.capture.get(cv2.CAP_PROP_FRAME_COUNT)

        # Set up codec and output video settings
        self.codec = cv2.VideoWriter_fourcc('M','J','P','G')
        self.output_video = cv2.VideoWriter(self.video_file_name, self.codec, 30, (self.frame_width, self.frame_height))

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (status, frame) = self.capture.read()
                if not self.q.empty():
                    try:
                        self.q.get_nowait()  # discard previous(unprocessed) frame
                    except queue.Empty:
                        pass
                self.q.put([status,frame])        

    def show_frame(self):
        # Display frames in main program
        if self.status:
            cv2.imshow(self.frame_name, self.frame)

        # Press Q on keyboard to stop recording
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            self.output_video.release()
            cv2.destroyAllWindows()
            exit(1)

    def save_frame(self):
        # Save obtained frame into video output file
        self.output_video.write(self.frame)

    def start_recording(self):
        # Create another thread to show/save frames
        def start_recording_thread():
            while True:
                try:
                    self.show_frame()
                    self.save_frame()
                except AttributeError:
                    pass
        self.recording_thread = Thread(target=start_recording_thread, args=())
        self.recording_thread.daemon = True
        self.recording_thread.start()
    
    def analyze_frame(self):
        pass

    def close(self):
        self.capture.release()
        self.output_video.release()
        cv2.destroyAllWindows()

    def viewCam(self):
        # read image in BGR format
        ret, image = self.capture.read()
        # convert image to RGB format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # get image infos
        height, width, channel = image.shape
        step = channel * width
        # create QImage from image
        qImg = QImage(image.data, width, height, step, QImage.Format_RGB888)
        # show image in img_label
        for camera in self.camera_labels:
            camera.setPixmap(QPixmap.fromImage(qImg))

