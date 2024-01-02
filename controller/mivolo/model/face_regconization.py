import time
import cv2
import mediapipe as mp
import numpy as np
import glob
import faiss
import pickle
import datetime
import threading
import os
from arcface import ArcFace
import face_recognition

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# from insightface.app import FaceAnalysis

MP_FACE_DETECTION = mp.solutions.face_detection
MP_DRAWING = mp.solutions.drawing_utils

# face_analysis = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# face_analysis.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

# face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

class FaceRecognition():
    def __init__(self, model_path, threshold = 0.5):
        self._face_detection = MP_FACE_DETECTION.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_emb = ArcFace.ArcFace()
        # self.face_feature = face_recognition()
        # self.feature = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.model_file_path = model_path
        self.encodings = []
        self.labels = []
        self.face_index = None
        self.threshold = 1.1
        self.load_DB()

    def get_face_box(self, image):
        """get the bounding box of human face

        Args:
            image: input image

        Returns:
            List: list bounding box of human face
        """
        try:
            results = self._face_detection.process(image)
            height, width, _ = image.shape
            # Draw face detections of each face.
            list_box = []
            if not results.detections:
                return list_box
            # annotated_image = image.copy()
            for detection in results.detections:
                x = int(detection.location_data.relative_bounding_box.xmin * width)
                y = int(detection.location_data.relative_bounding_box.ymin * height)
                w = int(detection.location_data.relative_bounding_box.width * width)
                h = int(detection.location_data.relative_bounding_box.height * height)
                list_box.append((x,y,w,h))

            return list_box
        except Exception as e:
            print(e)

    def get_face_box_face_recognition(self, image):
        try:
            imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = np.array(imgrgb)
            # detect face
            img_location = face_recognition.face_locations(img)
            list_box = []
            if len(img_location) > 0:
                for (top,right,bottom,left) in img_location:
                    list_box.append((left,top,right-left,bottom-top))
            return list_box
        except Exception as e:
            print(e)

    # def get_face_box_insightface(self, image):
    #     try:
    #         imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #         img = np.array(imgrgb)
    #         # detect face
    #         faces = face_analysis.get(img)
    #         list_box = []
    #         if len(faces) > 0:
    #             for i in range(len(faces)):
    #                 face = faces[i]
    #                 box = face.bbox.astype(int)
    #                 list_box.append((box[0],box[1],box[2]-box[0],box[3]-box[1]))
    #         # if len(faces) > 0:
    #         #     for (top,right,bottom,left) in faces:
    #         #         list_box.append((left,top,right-left,bottom-top))
    #         return list_box
    #     except Exception as e:
    #         print(e)


    def face_recog(self, face):
        """face recognition using facebook ai search

        Args:
            face (list): bounding box

        Returns:
            string: label result of face recognition
        """
        # global label
        t = datetime.datetime.now()
        label = 'unknown'
        i = []
        try:
            # face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (128,128))
            face_encode = self.face_emb.calc_emb(face)
            face_encode = np.array(face_encode,dtype=np.float32).reshape(-1,512)
            # print(face_encode)
            i,result = self.face_index.search(face_encode, k=1)
            if i[0][0] > self.threshold:
                print(i)
                return label,0
            label = [self.labels[i] for i in result[0]][0]
            print(f"{label}: {i}")
        except Exception as e:
            return 'None'
            pass

        # score = 1.1 - i[0][0]
        # if score < 0:
        #     score = 0
        return label
    
    def face_recognition_lib(self, face):
        """face recognition using facebook ai search

        Args:
            face (list): bounding box

        Returns:
            string: label result of face recognition
        """
        # global label
        t = datetime.datetime.now()
        label = 'unknown'
        i = []
        try:
            # face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (128,128))
            face_encode = self.face_feature.face_encodings(face)
            # face_encode = np.array(face_encode,dtype=np.float32).reshape(-1,512)
            # print(face_encode)
            i,result = self.face_index.search(face_encode, k=1)
            if i[0][0] > self.threshold:
                print(i)
                return label,0
            label = [self.labels[i] for i in result[0]][0]
        except Exception as e:
            return 'None'
            pass

        # score = 1.1 - i[0][0]
        # if score < 0:
        #     score = 0
        return label

    def train(self, image, label):
        """training new face & put to db file

        Args:
            image (Image): face image
            label (string): label of the image
        """

        db_file = open(self.model_file_path, "rb+")
        try:
            list_box = self.get_face_box(image)
            if len(list_box) > 0:
                x,y,w,h = self.get_bigest_box(list_box)
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (128,128))
                face_encode = self.face_emb.calc_emb(face)
                if len(face_encode) > 0:
                    if type(self.encodings) == list:
                        self.encodings.append(face_encode)
                    else:
                        self.encodings = np.append(self.encodings,[face_encode], axis=0)
                    # print(self.encodings)
                    self.labels.append(label)
            self.encodings = np.array(self.encodings,dtype=np.float32)
            db_file.seek(0)
            db_file.truncate()
            pickle.dump((self.encodings,self.labels), db_file)
            db_file.close()
            print(len(self.encodings))
            print(self.labels)
            print(f"trained {label}")
            return True
        except Exception as e:
            print(e)
            return False


    def train_a(self, image, label):
        """training new face & put to db file

        Args:
            image (Image): face image
            label (string): label of the image
        """

        db_file = open(self.model_file_path, "rb+")
        try:
            list_box = self.get_face_box(image)
            if len(list_box) > 0:
                x,y,w,h = self.get_bigest_box(list_box)
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (128,128))
                face_encode = self.face_emb.calc_emb(face)
                if len(face_encode) > 0:
                    if type(self.encodings) == list:
                        self.encodings.append(face_encode)
                    else:
                        self.encodings = np.append(self.encodings,[face_encode], axis=0)
                    # print(self.encodings)
                    self.labels.append(label)
            self.encodings = np.array(self.encodings,dtype=np.float32)
            db_file.seek(0)
            db_file.truncate()
            pickle.dump((self.encodings,self.labels), db_file)
            db_file.close()
            print(len(self.encodings))
            print(self.labels)
            print(f"trained {label}")
            return True
        except Exception as e:
            print(e)
            return False

    def load_DB(self):
        """load list face encode and labels, then put to faiss network

        Args:
            model_path (string): path of ther model file
        """
        # self.encodings = []
        # self.labels = []
        # begin = datetime.datetime.now()
        try:
            self.encodings = []
            self.labels = []
            if not os.path.isfile(self.model_file_path):
                with open(self.model_file_path, 'w') as f: pass
            else:
                try:
                    file = open(self.model_file_path,'rb')
                    object_file = pickle.load(file)
                    self.encodings = object_file[0]
                    self.labels = object_file[1]
                    print(f"labels: {self.labels}")
                    file.close()
                except: pass
            self.face_index = faiss.IndexFlatL2(512)
            # add vector
            if len(self.encodings) > 0:
                # print(self.encodings)
                self.face_index.add(self.encodings)
        except Exception as e:
            print(e)

    def delete(self, label):
        try:
            if label not in self.labels:
                print(f'{label} does not exist in dataset')
                return
            indices = [i for i, x in enumerate(self.labels) if x == label]
            # for idx in indices:
            if type(self.encodings) == list:
                for idx in sorted(indices, reverse=True):
                    del self.encodings[idx]
            else:
                self.encodings = np.delete(self.encodings, indices, axis=0)
            if type(self.labels) == list:
                for idx in sorted(indices, reverse=True):
                    del self.labels[idx]
            else:
                self.labels = np.delete(self.labels, indices)
            print(f'encodings: {len(self.encodings)}')
            print(f'labels: {len(self.labels)}')
            db_file = open(self.model_file_path, "rb+")
            db_file.seek(0)
            db_file.truncate()
            pickle.dump((self.encodings,self.labels), db_file)
            db_file.close()
            print(f"delete {label} from dataset")
            return True
        except Exception as e:
            print('delete label fail', e)
            return False


    def get_bigest_box(self, listbox):
        """get the biggest bouding box in list face box

        Args:
            listbox (list): list face box

        Returns:
            list: the biggest bounding box
        """
        index = 0
        max_S = 0
        for i in range(len(listbox)):
            current_S = listbox[i][2]*listbox[i][3]
            if current_S > max_S:
                index = i
                max_S = current_S

        return listbox[index]
    
    def draw_target(self,img, box):
        x1 = box[0]
        y1 = box[1]
        x2 = box[0]
        y2 = box[1] + box[3]
        x3 = box[0] + box[2]
        y3 = box[1]
        x4 = box[0] + box[2]
        y4 = box[1] + box[3]
        line_length = int(box[2]/5)

        cv2.circle(img, (x1, y1), 3, (255, 0, 255), -1)    #-- top_left
        cv2.circle(img, (x2, y2), 3, (255, 0, 255), -1)    #-- bottom-left
        cv2.circle(img, (x3, y3), 3, (255, 0, 255), -1)    #-- top-right
        cv2.circle(img, (x4, y4), 3, (255, 0, 255), -1)    #-- bottom-right

        cv2.line(img, (x1, y1), (x1 , y1 + line_length), (0, 255, 0), 2)  #-- top-left
        cv2.line(img, (x1, y1), (x1 + line_length , y1), (0, 255, 0), 2)

        cv2.line(img, (x2, y2), (x2 , y2 - line_length), (0, 255, 0), 2)  #-- bottom-left
        cv2.line(img, (x2, y2), (x2 + line_length , y2), (0, 255, 0), 2)

        cv2.line(img, (x3, y3), (x3 - line_length, y3), (0, 255, 0), 2)  #-- top-right
        cv2.line(img, (x3, y3), (x3, y3 + line_length), (0, 255, 0), 2)

        cv2.line(img, (x4, y4), (x4 , y4 - line_length), (0, 255, 0), 2)  #-- bottom-right
        cv2.line(img, (x4, y4), (x4 - line_length , y4), (0, 255, 0), 2)
        return img
    
    def mask_detection(self,face, model):
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (64, 64))
        face = img_to_array(face)
        face = preprocess_input(face)
        face = np.expand_dims(face, axis=0)

        mask = model.predict(face)[0]

        label = "Mask" if mask < 0.5 else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        return label, color



class CamThread_FaceRecog(threading.Thread):
    def __init__(self, previewName, camID, model_path):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
        self.model = model_path
    def run(self):
        print("Starting " + self.previewName)
        self.camPreview()

    def camPreview(self):
        cv2.namedWindow(self.previewName)
        cam = cv2.VideoCapture(self.camID)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        if cam.isOpened():
            rval, frame = cam.read()
        else:
            rval = False

        i = 0
        pre_lb = ''
        while rval:
            i += 1
            try:
                list_box= face_recognition.get_face_box(frame)
                # if isface:
                if len(list_box) > 0:
                    x,y,w,h = face_recognition.get_bigest_box(list_box)
                    # for (x,y,w,h) in list_box:
                    face = frame[y:y+h, x:x+w]
                    if i%60==0:
                        label = face_recognition.face_recog(face)
                        if label != pre_lb:
                            # thread = threading.Thread(target=welcomespeech, args=[label,])
                            # thread.start()
                            pre_lb = label
                    frame = draw_target(frame, (x,y,w,h))
                    frame = cv2.putText(frame, f"{label}", (x,y), cv2.FONT_HERSHEY_DUPLEX, 1, (62, 62, 250), 1, cv2.LSD_REFINE_STD)

            except: pass
            cv2.imshow(self.previewName, frame)
            rval, frame = cam.read()
            key = cv2.waitKey(20)
            if key == 27:  # exit on ESC
                break
        cv2.destroyWindow(self.previewName)

if __name__ == "__main__":
    pass