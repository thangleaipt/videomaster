import logging
import shutil
import threading
import time

from datetime import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import re
import queue

from server.machine_learning.report import send_report_to_db
from mivolo.person_model import PersonModel

from collections import Counter
import psutil
from unidecode import unidecode
import cv2
from server.config import DATETIME_FORMAT, STATIC_FOLDER, CAMERAS_RECORDS_FOLDER, DATASET_PEOPLE_FOLDER
from server.config import TELEGRAM_TOKEN
from server.config import DATE_TIME_FORMAT, DATETIME_FORMAT, STATIC_FOLDER
from scipy.optimize import linear_sum_assignment
import numpy as np

import telepot

from threading import Thread

from server.extension import db_session

from server.utils import send_email, telegram_send_text_message

from server.models import Camera, Record, Group, FeatureAnalyze, Feature, Telegram, People, Avatar
from server.machine_learning.Face_recognition.analyze_video_insightface import FaceAnalysisInsightFace
from mivolo.predictor import Predictor
from server.ma_schemas import AvatarsSchema

from sqlalchemy import and_

from ultralytics.yolo.utils.plotting import Annotator

avatars_schema = AvatarsSchema(many=True)

MAX_BUFFER_SIZE = 50

class CameraAnalyzer(Thread):
  def __init__(self, camera):
    super().__init__()
    self.camera = camera
    self.camera_status = True
    self.name_camera = camera.name
    self.rtsp = camera.rtsp
    self.features_ids = []

    self.predictor = Predictor()
    self.insightface = FaceAnalysisInsightFace()

    time_start_init_camera = time.time()
    self.cap = cv2.VideoCapture(self.rtsp)
    time_end_init_camera = time.time()

    # telegram
    self.telegrams = self.get_telegrams()    
    # send telegram
    self.telegram_buffer = queue.Queue()
    self.thread_telegram = threading.Thread(target=self.telegram_send_img_message)
    self.thread_telegram.start()

    self.status_update = False

    self.list_total_id = []
    self.list_person_model = []
    self.list_id_check_in_frame = []


    self.list_label_greeting = []

    self.list_person_in_roi = []
    self.list_name_in_roi = []
    self.index_exception = 0

    self.bot = telepot.Bot(TELEGRAM_TOKEN) 

    if f"{self.rtsp}".isdigit():
      self.rtsp = int(self.rtsp)

    self.index_frame = 0
 

    if time_end_init_camera - time_start_init_camera > 50:
      self.camera_status = False
      self.cap.release()
      text_status = f"Thời gian: {time.strftime(DATETIME_FORMAT)} \n Camera {self.rtsp} không hoạt động. Vui lòng kiểm tra lại"
      telegram_send_text_message(text_status, self.telegrams)

      send_email("thangxajk@gmail.com",f"SỰ CỐ CAMERA {self.rtsp}",text_status)
        
    self.frame_buffer = None

    self.camera_id = camera.id
    self.start_record = None
    self.record_status = False
    self.group_name = self.get_group_name()

    self.list_person_db = []

    # biên để lưu record camera
    self.record_extension = "mp4"
    self.fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    self.camera_fps = self.cap.get(cv2.CAP_PROP_FPS)
    self.frame_width_writer = 800
    self.frame_height_writer = 600
    self.video_writer = None

  def get_group_name(self):
    session = db_session()
    group = session.query(Group).filter(Group.id == self.camera.group_id).first()
    session.close()

    return f"{group.name}".replace(" ", "_")

  def get_telegrams(self):
    session = db_session()
    telegrams = session.query(Telegram).filter(Telegram.status == 1).all()
    session.close()

    return telegrams

  def extract_ip_from_rtsp(self):
    if f"{self.rtsp}".isdigit():
      return f"{self.rtsp}"
    
    ip_matches = re.findall(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', self.rtsp)
    if ip_matches:
      ip = ip_matches[0]
      return ip
    
    else:
      return None
      
  def get_output_record_path(self):
    now = datetime.now()

    group_folder = os.path.join(CAMERAS_RECORDS_FOLDER, self.group_name.lower())
    ip_folder = os.path.join(group_folder, self.extract_ip_from_rtsp())
    year_folder = os.path.join(ip_folder, f"nam_{now.year}")
    month_folder = os.path.join(year_folder, f"thang_{now.month}")
    day_folder = os.path.join(month_folder, f'ngay_{now.day}')
    hour_folder = os.path.join(day_folder, f"{now.hour}_gio")
    filename = os.path.join(hour_folder, f"phut_{int(now.timestamp())}.{self.record_extension}")

    if not os.path.exists(STATIC_FOLDER):
      os.mkdir(STATIC_FOLDER)

    if not os.path.exists(CAMERAS_RECORDS_FOLDER):
      os.mkdir(CAMERAS_RECORDS_FOLDER)

    if not os.path.exists(group_folder):
      os.mkdir(group_folder)
      
    if not os.path.exists(ip_folder):
      os.mkdir(ip_folder)

    if not os.path.exists(year_folder):
      os.mkdir(year_folder)

    if not os.path.exists(month_folder):
      os.mkdir(month_folder)

    if not os.path.exists(day_folder):
      os.mkdir(day_folder)

    if not os.path.exists(hour_folder):
      os.mkdir(hour_folder)
    
    return filename

  def update_record_end_time(self):
    session = db_session()

    # lấy ra thông tin bản ghi vừa lưu
    record = session.query(Record)\
      .filter(and_(Record.camera_id == self.camera_id, Record.start == self.start_record)).first()
    
    session.query(Record).filter(Record.id == record.id).update({
      Record.end: datetime.now().strftime(DATETIME_FORMAT)
    })
    
    session.commit()

  def save_record_to_db(self):
    session = db_session()
    record = Record(self.start_record, None, self.get_output_record_path(), self.camera_id, None)
    session.add(record)
    session.commit()
    session.close()

  def remove_old_record(self):
    session = db_session()
    records = session.query(Record).order_by(Record.start).all()
    count = 0
    number_remove = 15

    for record in records:
      if count >= number_remove:
        break

      if os.path.exists(record.path):
        count += 1
        os.remove(record.path)

    session.close()

  def set_record_status(self, new_status):
    self.record_status = new_status
    session = db_session()

    cam = session.query(Camera).filter(Camera.id == self.camera_id)
    if new_status:
      cam.update({Camera.record_status: 1})
    else:
      cam.update({Camera.record_status: 0})

    session.commit()

  def handleRecording(self, frame):
    if self.record_status and frame is not None:
      if self.video_writer is None:
        self.video_writer = cv2.VideoWriter(self.get_output_record_path(), self.fourcc, self.camera_fps, (self.frame_width_writer, self.frame_height_writer))

      disk_usage = psutil.disk_usage("./")
      disk_free = disk_usage.free / (1024**3)

      frame = cv2.resize(frame, (self.frame_width_writer, self.frame_height_writer))
      self.video_writer.write(frame)

      if datetime.now().second != 0 and self.start_record is None:
        if disk_free < 0.5:
          self.remove_old_record()

        self.start_record = datetime.now().strftime(DATETIME_FORMAT)
        self.save_record_to_db()

      if datetime.now().minute % 2 == 0 and datetime.now().second == 0 and self.start_record is not None:
        # lưu bản ghi cũ
        self.video_writer.release()
        self.update_record_end_time()

        # bắt đầu bản ghi mới
        self.video_writer = None
        self.start_record = None

    if not self.record_status:
      if self.start_record is not None:
        self.update_record_end_time()
        self.video_writer.release()
        self.start_record = None
        self.video_writer = None

  def get_group_dataset_folder(self):
    session = db_session()  

    group_name = session.query(Group).filter(Group.id == self.camera.group_id).first().name
    group_name = unidecode(f"{group_name}".replace(" ", ""))
    group_dataset_folder = os.path.join(DATASET_PEOPLE_FOLDER, f"{self.camera.group_id}_{group_name}".lower()) 
    
    session.close()
    return group_dataset_folder

  def camera_query_people(self):
    session = db_session()  
    people = session.query(
      People.id,
      People.name,
      People.age,
      People.type,
      People.gender
    ).filter(
      People.group_id == self.camera.group_id,
      People.status == 1
    ).all()

    for i, person in enumerate(people):
      person_name = unidecode(f"{person.name}".replace(" ", ""))
      avatars = session.query(Avatar).filter(Avatar.people_id == person.id).all()

      group_dataset_folder = self.get_group_dataset_folder()
      people_dataset_folder = os.path.join(group_dataset_folder, f"{person.id}_{person_name}".lower())
      
      people[i] = person._asdict()
      people[i]['avatars'] = avatars_schema.dump(avatars)
      people[i]['group_dataset_folder'] = group_dataset_folder
      people[i]['people_dataset_folder'] = people_dataset_folder
      
    session.close()
    return people

  def train(self, person_delete):
    self.insightface.list_person_trained = self.camera_query_people()
    self.insightface.person_delete = person_delete
    if person_delete is not None:
      shutil.rmtree(person_delete[0]['people_dataset_folder'])
    self.insightface.load_db_from_database()
    print(f"Camera: {self.camera.name} trained !")
  
  def get_features_id(self):
    session = db_session()

    features = session.query(Feature).join(
      FeatureAnalyze, Feature.id == FeatureAnalyze.feature_id
    ).filter(
      FeatureAnalyze.camera_id == self.camera_id
    ).all()
    session.close()
    
    self.features_ids = []

    for feature in features:
      if feature.id not in self.features_ids:
        self.features_ids.append(feature.id)

  def telegram_send_img_message(self):
    try:
      # for telegram in telegrams:
        while True:
          img_path, person_model, telegrams = self.telegram_buffer.get()
          print(f"Telegram send image: {img_path}")
          text_send = f"*Camera: {self.name_camera}*\n"
          text_send += f"ID: _{person_model.id}_\n"
      
          if person_model.label_name is not None and person_model.label_name != "":
              text_send += f"Tên: *{person_model.label_name}*\n"
          else:
              if person_model.role == 1:
                text_send += f"Tên: Người không xác định\n"
              elif person_model.role == 0:
                text_send += f"Nhân viên: Người lạ\n"

          if person_model.average_check_mask is not None and person_model.average_check_mask == 1:
              text_send += f"Đeo khẩu trang: _Có_\n"
          elif person_model.average_check_mask is not None and person_model.average_check_mask == 0:
              text_send += f"Đeo khẩu trang: _Không_\n"
          else:
              text_send += f"Đeo khẩu trang: _Không xác định_\n"
          if person_model.average_age is not None:
              text_send += f"Tuổi: _{int(person_model.start_age)}_ - _{int(person_model.end_age)}_\n"
          else:
              text_send += f"Tuổi: _Không xác định_\n"
          if person_model.average_gender != "" and person_model.average_gender is not None:
              if person_model.average_gender is not None:
                  gender = person_model.average_gender
              if gender == "male" or gender[0] == "male":
                  text_send += f"Giới tính: _Nam_\n"
              elif gender == "female" or gender[0] == "female":
                  text_send += f"Giới tính: _Nữ_\n"
          else:
              text_send += f"Giới tính: _Không xác định_\n"
          if person_model.name_color != "" and person_model.name_color is not None:
              text_send += f"Màu trang phục: _{person_model.name_color}_\n"
          else:
              text_send += f"Màu trang phục: _Không xác định_\n"

          text_send += f"Thời gian nhận diện: _{datetime.now().strftime(DATETIME_FORMAT)}_\n"

          for telegram in telegrams:
            try:
              time_start = time.time()
              self.bot.sendPhoto(telegram.chat_id, photo=open(img_path, 'rb'), caption=text_send, parse_mode= 'Markdown')
              time_end = time.time()
              logging.info(f"time telegram: {time_end - time_start}s")
            except Exception as ex:
              print(f"Exception[send_telegram_message]: {ex}")
              logging.error(f"Exception[send_telegram_message]: {ex}")
          self.telegram_buffer.task_done()
    except Exception as ex:
      print(f"Exception[telegram_send_img_message]: {ex}")
      logging.error(f"Exception[telegram_send_img_message]: {ex}")

  def run(self):
    try:
      self.telegrams = self.get_telegrams()
      self.train(None)
      self.get_features_id()

      while True:
        success, frame = self.cap.read()

        if not success:
          print(f" Camera {self.rtsp} image is None")
          self.cap.release()
          time_start_init = time.time()
          self.cap = cv2.VideoCapture(self.rtsp)
          time_end_init = time.time()
          if time_end_init - time_start_init > 50:
            self.camera_status = False
            self.cap.release()
            text_status = f"Thời gian: {time.strftime(DATETIME_FORMAT)} \n Camera {self.rtsp} không hoạt động. Vui lòng kiểm tra lại"
            telegram_send_text_message(text_status, self.telegrams)
              
            send_email("thangxajk@gmail.com",f"SỰ CỐ CAMERA {self.rtsp}",text_status)
            break
          if not self.cap.isOpened():
              print("Error: Could not reopen camera.")
              break
          continue
             
        self.index_frame += 1
        self.handleRecording(frame)
        frame = self.analyze_video(frame)
        self.frame_buffer = frame
        # if self.frame_buffer.qsize() >= MAX_BUFFER_SIZE:
        #   self.frame_buffer = queue.Queue(maxsize=MAX_BUFFER_SIZE)
        # self.frame_buffer.put(frame)
        
    except Exception as e:
      print(f"[CameraAnalyzer][run]: {e}")
  
  def gen_frames(self):
    try:
      while True:
        frame = self.frame_buffer
        if frame is not None:
          _, buffer = cv2.imencode('.jpg', frame)
          # print(f"data {self.camera_id}: {len(bytearray(buffer))}")
          yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(buffer) + b'\r\n')
          # self.frame_buffer.task_done()
    except Exception as e:
      print(f"[CameraAnalyzer][get_frames_webcam]: {e}")

  def match_face_to_person(self,person_boxes, face_boxes):
    num_persons = len(person_boxes)
    num_faces = len(face_boxes)

    # Tạo ma trận chứa giá trị IoU giữa tất cả các cặp hộp người và khuôn mặt
    iou_matrix = np.zeros((num_persons, num_faces))

    for i in range(num_persons):
        for j in range(num_faces):
            iou_matrix[i, j] = self.calculate_iou(person_boxes[i][0][:4], face_boxes[j][0])

    # Áp dụng thuật toán Hungarian để ghép cặp
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)

    # Trả về danh sách các cặp được ghép
    matched_pairs = [(row, col) for row, col in zip(row_ind, col_ind)]

    return matched_pairs

  def get_matched_pairs(self,person_boxes, face_boxes):
      matched_pairs = self.match_face_to_person(person_boxes, face_boxes)

      # Tạo danh sách mới chứa tất cả các cặp face và body
      all_pairs = []

      for person_idx, face_idx in matched_pairs:
          all_pairs.append((person_boxes[person_idx], face_boxes[face_idx]))

      # Kiểm tra nếu có face nào không được match, thêm vào danh sách với giá trị None cho body
      unmatched_faces = set(range(len(face_boxes))) - set(col for _, col in matched_pairs)

      for face_idx in unmatched_faces:
          all_pairs.append((None, face_boxes[face_idx]))

      # Kiểm tra nếu có body nào không được match, thêm vào danh sách với giá trị None cho face
      unmatched_bodies = set(range(len(person_boxes))) - set(row for row, _ in matched_pairs)

      for person_idx in unmatched_bodies:
          all_pairs.append((person_boxes[person_idx], None))

      return all_pairs

  def calculate_iou(self,box1, box2):
          try:
              x1, y1, w1, h1 = box1[0], box1[1], box1[2]-box1[0], box1[3]-box1[1]
              x2, y2, w2, h2 = box2[0], box2[1], box2[2], box2[3]
              # x1, y1, w1, h1 = box1
              # x2, y2, w2, h2 = box2

              x1_left, x1_right = x1, x1 + w1
              y1_top, y1_bottom = y1, y1 + h1
              x2_left, x2_right = x2, x2 + w2
              y2_top, y2_bottom = y2, y2 + h2

              x_intersection = max(0, min(x1_right, x2_right) - max(x1_left, x2_left))
              y_intersection = max(0, min(y1_bottom, y2_bottom) - max(y1_top, y2_top))
              intersection_area = x_intersection * y_intersection

              area1 = w1 * h1
              area2 = w2 * h2
              union_area = area1 + area2 - intersection_area

              iou = intersection_area / union_area

              return iou
          except Exception as e:
              print(f"[analyze_video_insightface][calculate_iou]: {e}")
              return 0
        
  def extend_image(self,frame, face_box):
    try:
        x, y, w, h = face_box
        frame_height, frame_width = frame.shape[:2]
        if w < 64:
            target_size = (128, 128)
        else:
            target_size = (int(w*2), int(h*2))  
        delta_w = target_size[0] - w
        delta_h = target_size[1] - h
        x1, y1 = int(x - delta_w // 2), int(y - delta_h // 2)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = int(min(frame_width, x1 + target_size[0]))
        y2 = int(min(frame_height, y1 + target_size[1]))
        expanded_face = frame[y1:y2, x1:x2]
        return expanded_face
    except Exception as e:
        print(f"[{datetime.now().strftime(DATETIME_FORMAT)}][analyze_video][extract_face]: ", e)
        return None
    
  def count_most_frequent_element(self,lst):
    if len(lst) == 0:
        return None
    
    element_count = Counter(lst)
    max_count = max(element_count.values())
    most_frequent_elements = [key for key, value in element_count.items() if value == max_count]
    
    return most_frequent_elements

  def average_number(self,age_list):
      if not age_list or age_list is None:  # Check if the list is empty to avoid division by zero
          return 0
      total_age = sum(age_list)
      average = total_age / len(age_list)
      
      return average
  

  def analyze_video(self, image):
    try:
        frame = image.copy()
      
        result_track = []
        logging.info(f"[analyze_video][time_detect]: Tracking start")
  
        result_track = self.predictor.analyze_tracking_boxmot(frame)

        # insightface recognition
        list_recognition_insightface = []
        
        if 1 in self.features_ids:
            logging.info(f"[analyze_video_insightface][time_insightface]: Insightface start")
            list_recognition_insightface = self.insightface.analyze_insightface_frame(frame)
            logging.info(f"[analyze_video_insightface][time_insightface]: Merge start")
            list_recognition_insightface_merge = self.get_matched_pairs(result_track, list_recognition_insightface)

        annotator = Annotator(
                frame,
                None,
                None,
                font="Arial.ttf",
                pil=False,
            )
        color_person = (0, 255, 0)
        color_text = (0, 0, 0)
        color_face = (255, 255, 0)

        box_face = []
        box_person = []

        for d, instance in list_recognition_insightface_merge:
            image_save = image.copy()
            extend_face_image = None
            person_image = None
            face_image = None
            
            label = ""
            path_dir_image = ""
            path_save_face_image = ""
            path_save_person_image = ""

            label_name = None
            label_mask = None

            guid = None
            id_db = None
            age = None
            gender = None
            main_color_clothes = None
            name_color = None


            if instance is not None and len(instance) > 0:
                box_face = instance[0]
                label_name = instance[1]
                id_db = instance[2]
                age = instance[4]
                gender = instance[5]
                label_mask = instance[6]

                box_face = [int(box_face[0]), int(box_face[1]), int(box_face[2]), int(box_face[3])]
                cv2.rectangle(image_save, (box_face[0], box_face[1]), (box_face[2]+box_face[0], box_face[3]+box_face[1]), (0, 0, 255), 2)
                face_image = image[max(box_face[1],0):min(box_face[3]+box_face[1], frame.shape[0]), max(box_face[0],0):min(box_face[2]+box_face[0], frame.shape[1])]
                box_face_plot = [box_face[0], box_face[1], box_face[2]+box_face[0], box_face[3]+box_face[1]]
                
                if face_image is None or face_image.shape[0] == 0 or face_image.shape[1] == 0:
                    face_image = None
                else:
                    extend_face_image = self.extend_image(image, box_face)
                if label_name is not None:
                  label_name = unidecode(label_name)
                  label = f" {label_name}"
                else:
                   label = ""
                label += f" {age:.1f}"
                if gender is not None:
                    if gender == "male":
                        label += "M"
                    else:
                        label += "F"

                if label_mask is not None:
                    if label_mask == 0:
                        label += f" No Mask"
                        color_face = (0, 0, 255)
                    elif label_mask == 1:
                        label += f" Mask"
                        color_face = (255, 0, 0)

                annotator.box_label(box_face_plot,"",color_face)

            if d is not None and len(d) > 0:
                box_person = d[0][0:4]
                guid = d[0][4]
                label += f" {guid}"
                main_color_clothes = d[1]
                name_color = d[2]
                if main_color_clothes is not None:
                    color_person = main_color_clothes
                    color_text = (255, 255, 255)

                cv2.rectangle(image_save, (box_person[0], box_person[1]), (box_person[2], box_person[3]), (0, 255, 0), 2)
                person_image = image[box_person[1]:box_person[3], box_person[0]:box_person[2]]
                annotator.box_label(box_person, label, color=color_person,txt_color=color_text)

            frame = annotator.result()

            if label_name is not None and face_image is not None:
                path_dir_image = f"{STATIC_FOLDER}/results/Camera_{self.camera_id}/{datetime.now().strftime(DATE_TIME_FORMAT)}/id_{label_name.replace(' ', '')}"
                path_save_face_image = f"{path_dir_image}/{time.time()}.jpg"
                if person_image is not None:
                    path_save_person_image = f"{path_dir_image}/person_{time.time()}.jpg"
            elif label_name is None:
                if face_image is not None or person_image is not None:
                    path_dir_image = f"{STATIC_FOLDER}/results/Camera_{self.camera_id}/{datetime.now().strftime(DATE_TIME_FORMAT)}/id_{guid}"
                if face_image is not None:
                    path_save_face_image = f"{path_dir_image}/{time.time()}.jpg"
                if person_image is not None:
                    path_save_person_image = f"{path_dir_image}/person_{time.time()}.jpg"

            if path_dir_image != "" and not os.path.exists(path_dir_image):
                os.makedirs(path_dir_image)
                print(f"Create folder {path_dir_image}")
            elif path_dir_image == "":
                path_save_face_image = ""
                path_save_person_image = ""

            path_image = f"{path_dir_image}/origin_{time.time()}.jpg"


            if guid not in self.list_total_id:
                person_model = PersonModel()
                # person model
                person_model.id = guid
                if label_name is not None:
                    person_model.list_face_name.append(label_name)
                    person_model.label_name = label_name
                
                # Mask
                if label_mask is not None:
                    person_model.list_check_masks.append(label_mask)
                    person_model.average_check_mask = label_mask

                # Age
                if age is not None:
                    person_model.list_age.append(age)
                    person_model.average_age = age

                # Gender
                if gender is not None:
                    person_model.list_gender.append(gender)
                    person_model.average_gender = gender
                
                # ID DB
                if id_db is not None and label_name is not None and extend_face_image is not None:
                    person_model.id_db = id_db

                # Main color
                if main_color_clothes is not None:
                    person_model.code_color = f"{main_color_clothes[0]},{main_color_clothes[1]},{main_color_clothes[2]}"
                if name_color is not None:
                    person_model.name_color = name_color

                person_model.list_image_path = []
                if extend_face_image is not None and path_save_face_image != "":
                    if extend_face_image.shape[0] > 0 and extend_face_image.shape[1] > 0:
                        cv2.imwrite(path_save_face_image, extend_face_image)
                        
                        if path_save_face_image not in person_model.list_image_path:
                            person_model.list_image_path.append(path_save_face_image)

                        if label_name is not None:
                            person_model.role = 2
                        else:
                            if box_face[2] < 50:
                              person_model.role = 1
                            else:
                              person_model.role = 0

                if person_image is not None and person_image.shape[0] > 0 and person_image.shape[1] > 0:
                        if extend_face_image is None:
                          person_model.role = 1
                        cv2.imwrite(path_save_person_image, person_image)
                        if path_save_person_image not in person_model.list_image_path:
                            person_model.list_image_path.append(path_save_person_image)

                cv2.imwrite(path_image, image_save)
                person_model.time = datetime.now().strftime(DATETIME_FORMAT)
                person_model.list_feature = self.features_ids
                self.list_person_model.append(person_model)
                self.list_total_id.append(guid)
            else:
                index = self.list_total_id.index(guid)
                if self.index_frame % 2 == 0:
                    if label_name is not None:
                        self.list_person_model[index].list_face_name.append(label_name)
                        if len(self.list_person_model[index].list_face_name) > 50:
                            del self.list_person_model[index].list_face_name[0]
                    if age is not None:
                        self.list_person_model[index].list_age.append(age)
                        if len(self.list_person_model[index].list_age) > 50:
                            del self.list_person_model[index].list_age[0]
                    
                    if gender is not None:
                        self.list_person_model[index].list_gender.append(gender)
                        if len(self.list_person_model[index].list_gender) > 50:
                            del self.list_person_model[index].list_gender[0]

                    if label_mask is not None:
                        self.list_person_model[index].list_check_masks.append(label_mask)
                        if len(self.list_person_model[index].list_check_masks) > 50:
                            del self.list_person_model[index].list_check_masks[0]

                if id_db is not None and extend_face_image is not None and label_name is not None:
                    self.list_person_model[index].id_db = id_db
                    
                self.list_person_model[index].counting_tracking += 1

                # Send telegram
                if (self.list_person_model[index].counting_tracking % 5 == 0 or self.list_person_model[index].counting_tracking == 1 or (self.list_person_model[index].label_name != label_name and label_name is not None)):
                    if label_name is not None:
                        self.list_person_model[index].label_name = label_name
                    if gender is not None:
                        self.list_person_model[index].average_gender = gender
                    else:
                        self.list_person_model[index].average_gender = self.count_most_frequent_element(self.list_person_model[index].list_gender)

                    self.list_person_model[index].average_age = self.average_number(self.list_person_model[index].list_age)
                    self.list_person_model[index].average_check_mask = label_mask
                    if len(self.list_person_model[index].list_age) > 0:
                        self.list_person_model[index].start_age = min(self.list_person_model[index].list_age)
                        self.list_person_model[index].end_age = max(self.list_person_model[index].list_age)
                    
                    if extend_face_image is not None and path_save_face_image != "":
                        if extend_face_image.shape[0] > 0 and extend_face_image.shape[1] > 0:
                            extend_face_image = cv2.resize(extend_face_image, (128, 128))
                            cv2.imwrite(path_save_face_image, extend_face_image)
                            if path_save_face_image not in self.list_person_model[index].list_image_path:
                                self.list_person_model[index].list_image_path.append(path_save_face_image)
                            if label_name is not None:
                                self.list_person_model[index].role = 2
                            else:
                                if box_face[2] < 50:
                                  self.list_person_model[index].role = 1
                                else:
                                  self.list_person_model[index].role = 0
                    if person_image is not None and person_image.shape[0] > 0 and person_image.shape[1] > 0:
                        if person_image.shape[1] > 128 and person_image.shape[1] > person_image.shape[0]:
                            person_image = cv2.resize(person_image, (128, int(person_image.shape[0] * 128 / person_image.shape[1])))
                        elif person_image.shape[0] > 128 and person_image.shape[1] <= person_image.shape[0]:
                            person_image = cv2.resize(person_image, (int(person_image.shape[1] * 128 / person_image.shape[0]),128))
                        cv2.imwrite(path_save_person_image, person_image)
                        if path_save_person_image not in self.list_person_model[index].list_image_path:
                            self.list_person_model[index].list_image_path.append(path_save_person_image)
                        if extend_face_image is None:
                            self.list_person_model[index].role = 1
                    cv2.imwrite(path_image, image_save)
                    if self.list_person_model[index].counting_telegram < 5:
                      self.telegram_buffer.put([path_image, self.list_person_model[index], self.telegrams])
                      self.list_person_model[index].counting_telegram += 1

                    print(f"Send telegram: {self.list_person_model[index].counting_tracking}, {self.list_person_model[index].counting_telegram}, Path Face: {path_save_face_image} Path Person: {path_save_person_image}")
                    logging.info(f"Send telegram: {self.list_person_model[index].counting_tracking}, {self.list_person_model[index].counting_telegram}, Path Face: {path_save_face_image} Path Person: {path_save_person_image}")
        
        # Send data to report
        if self.index_frame % 100 == 0:
            logging.info(f"[analyze_video][index_frame] Send data to report: {self.index_frame}, {len(self.list_person_model)}")
            if len(self.list_person_model) > 0:
                send_report_to_db(self)
                
        if len(self.list_person_model) >= 1000:      
            self.list_total_id.remove(self.list_total_id[0])
            self.list_person_model.remove(self.list_person_model[0])             
        return frame
    except Exception as e:
        print(f'[{datetime.now().strftime(DATETIME_FORMAT)}][analyze][generate_frames]: {e}')
        logging.exception(f'[analyze][generate_frames]: {e}')
        return image