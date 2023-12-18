import cv2
import time
from threading import Thread


class VideoStreamWidget(object):
    def __init__(self, src=0):
        self.src = src
        self.capture = cv2.VideoCapture(src)
        self.status = None
        self.frame = None
        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()
            if not self.status or self.frame is None:
                print("Error: Could not read frame. Reopening the camera...")
                self.capture.release()  # Release the camera
                self.capture = VideoStreamWidget(self.src)  # Reopen the camera
                if not self.capture.status:
                    print("Error: Could not reopen camera.")
                    # break
                continue

    def release(self):
        # Release the video capture when you're done
        self.capture.release()
    
    def show_frame(self):
        # Display frames in main program
        cv2.imshow('frame', self.frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            self.capture.release()
            cv2.destroyAllWindows()
            exit(1)
