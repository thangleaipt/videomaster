import time
import cv2
import numpy as np
import glob
import faiss
import pickle
import datetime
import threading
import os
from arcface import ArcFace
import face_recognition
from gtts import gTTS
# from insightface.app import FaceAnalysis

# MP_FACE_DETECTION = mp.solutions.face_detection
# MP_DRAWING = mp.solutions.drawing_utils

# face_analysis = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

# face_analysis.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

# face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

class FaceRecognition():
    def __init__(self, model_path, threshold = 250):
        # self._face_detection = MP_FACE_DETECTION.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_emb = ArcFace.ArcFace()
        # self.feature = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        # self.feature.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))
        self.model_file_path = model_path
        self.encodings = []
        self.labels = []
        self.face_index = None
        self.threshold = 250
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

    def get_face_box_dlib(self, image):
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

    def get_face_box_insightface(self, image):
        try:
            imgrgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img = np.array(imgrgb)
            # detect face
            faces = self.feature.get(img)
            list_box = []
            if len(faces) > 0:
                for i in range(len(faces)):
                    face = faces[i]
                    box = face.bbox.astype(int)
                    list_box.append((box[0],box[1],box[2]-box[0],box[3]-box[1]))
            # if len(faces) > 0:
            #     for (top,right,bottom,left) in faces:
            #         list_box.append((left,top,right-left,bottom-top))
            return faces
        except Exception as e:
            print(e)


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
            # face = cv2.resize(face, (128,128))
            face_encode = face.embedding
            face_encode = np.array(face_encode,dtype=np.float32).reshape(-1,512)
            # print(face_encode)
            i,result = self.face_index.search(face_encode, k=1)
            if i[0][0] > self.threshold:
                return label,0
            # else:
            print(i)
            # print(result)
            label = [self.labels[i] for i in result[0]][0]
        except Exception as e:
            return 'None'
            pass
        # print(f'search finish in: {datetime.datetime.now() - t}')\
        # label = checklabel(label)
        score = 1.1 - i[0][0]
        if score < 0:
            score = 0
        return label, score*100

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


    def train(self, image, label):
        """training new face & put to db file

        Args:
            image (Image): face image
            label (string): label of the image
        """

        db_file = open(self.model_file_path, "rb+")
        try:
            list_box = self.get_face_box_insightface(image)
            if len(list_box) > 0:
                # x,y,w,h = self.get_bigest_box(list_box)
                x,y,w,h = int(list_box[0].bbox[0]),int(list_box[0].bbox[1]),int(list_box[0].bbox[2]),int(list_box[0].bbox[3])
                # face = image[y:y+h, x:x+w]
                # face = cv2.resize(face, (128,128))
                face_encode = list_box[0].embedding
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
                    print(f"encodings: {len(self.encodings)}")
                    self.labels = object_file[1]
                    print(f"labels: {len(self.labels)}")
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


def draw_target(img, box):
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

if __name__ == "__main__":
    model_path = r"G:\Ai-Cam-Viewer\model\models\face_recog\encoding_features_training.pkl"
    face_recog = FaceRecognition(model_path)
    # # # # face_recog.delete('hoang')

    # cap = cv2.VideoCapture(0)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
    # # savefolder = 'fuji_result'

    # while cap.isOpened():
    # # for img in glob.glob(r"D:\backups3\ai-cam\fujimart_17_10\*.jpg"):
    #     # img = r"C:\Users\VMIO\Desktop\1666446155.jpg"
    #     ret, frame = cap.read()
    #     # frame = cv2.imread(img)
    #     # name = os.path.basename(img)
    #     # heigh,width,_ = frame.shape
    #     list_box = face_recog.get_face_box_dlib(frame)
    #     print(list_box)
    #     print(len(list_box))
    #     if len(list_box) > 0:
    #         for (x,y,w,h) in list_box:
    #             face = frame[y:y+h, x:x+w]
    #             # if i%60==0:
    #             label = face_recog.face_recog(face)
    #             frame = draw_target(frame, (x,y,w,h))
    #             frame = cv2.putText(frame, f"{label}", (x,y), cv2.FONT_HERSHEY_DUPLEX, 1, (62, 62, 255), 1, cv2.LSD_REFINE_STD)
    #             # cv2.imwrite(os.path.join(savefolder,f'{name}_out.jpg'),frame)
    #     # frame = cv2.resize(frame, (int(width/2),int(heigh/2)))
    #     cv2.imshow('image', frame)

    #     # cv2.waitKey(0)
    #     if cv2.waitKey(100) & 0xff == ord('q'):
    #         break
    # cap.release()
    # cv2.destroyAllWindows()


    # cam0 = CamThread_FaceRecog("cam 0", 0, model_path)
    # # cam1 = CamThread_FaceRecog("cam 1", 1, model_path)
    # cam0.start()
    # # cam1.start()

    # cam0.join()
    # # cam1.join()

    # # Destroy all the windows
    # cv2.destroyAllWindows()

    ##train folder
    # for imgpath in glob.glob(r"D:\AI_project\Mio-Ai-Cam-Viewer\model\models\face_recog\dataset\Duong Van Thang\*.jpg"):
    #     # print(path)
    #     # imgpath = r"D:\AI_project\Face-Recognition-with-Faiss\dataset\a phong\phong1.jpg"

    #     img = cv2.imread(imgpath)
    #     # cv2.imshow('image', img)
    #     # cv2.waitKey(0)
    #     lb = os.path.dirname(imgpath).split('\\')[-1]
    #     # lb = os.path.basename(imgpath).split('.')[0]
    #     # face_recog = FaceRecognition(model_path=model_path)
    #     face_recog.train(image=img,label=lb)

    #train 1 image
    list_lb = []
    imgpath = r"C:\Users\Thangxajk\Downloads\photo_6339327151886677642_y.jpg"
    frame = cv2.imread(imgpath)
    lb = os.path.dirname(imgpath).split('\\')[-1]
    # face_recog = FaceRecognition(model_path=model_path)
    list_box = face_recog.get_face_box_insightface(frame)
    print(len(list_box))
    if len(list_box) > 0:
        for face in list_box:
            x,y,w,h = int(face.bbox[0]),int(face.bbox[1]),int(face.bbox[2])-int(face.bbox[0]),int(face.bbox[3])-int(face.bbox[1])
            # face = frame[y:y+h, x:x+w]
            # if i%60==0:
            label, score = face_recog.face_recog(face)
            if label == 'unknown':
                color_lb = (255,0,0)
            elif label == 'None':
                color_lb = (87, 86, 86)
            else:
                color_lb = (0,0,255)
            frame = draw_target(frame, (x,y,w,h))
            w,h = cv2.getTextSize(f"{label} [{int(score)}%]",cv2.FONT_HERSHEY_DUPLEX,0.7,2)[0]
            cv2.rectangle(frame,(x,y-5),(x+w,y-h-10),color_lb,-1)
            frame = cv2.putText(frame, f"{label} ", (x,y-10), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2, cv2.LSD_REFINE_STD)
            cv2.imshow('image', frame)
            cv2.waitKey(0)

    else:
        list_lb = []

    cv2.destroyAllWindows()
