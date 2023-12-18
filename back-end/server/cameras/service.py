import logging
import shutil
import threading
import time

from server.extension import db_session
from server.models import Camera, Record, Group, User, CamerasUser, FeatureAnalyze, Feature, Telegram, People, Avatar
from server.ma_schemas import CameraSchema, CommonSchema, AvatarsSchema
from flask import Response, request, jsonify
from threading import Thread
from sqlalchemy import and_, or_
import cv2
from flask_jwt_extended import get_jwt_identity
from datetime import datetime
import os
import re
import numpy as np
import queue
import psutil
from unidecode import unidecode

from server.models import Camera, Feature
from server.config import DATETIME_FORMAT, STATIC_FOLDER, CAMERAS_RECORDS_FOLDER, DATASET_PEOPLE_FOLDER

from server.config import SENDER_EMAIL, PASSWORD_EMAIL, SMTP_SERVER, SMTP_PORT, TELEGRAM_TOKEN

cameras_schema = CameraSchema(many=True)
common_schema = CommonSchema(many=True)
avatars_schema = AvatarsSchema(many=True)

threads_camera_analyzer = {}
MAX_BUFFER_SIZE = 1000

class CameraAnalyzer(Thread):
  def __init__(self, camera):
    super().__init__()
    self.camera = camera
    self.name_camera = camera.name
    self.rtsp = self.camera.rtsp  
    self.features_ids = []

    self.outputFrame = []

    time_start_init_camera = time.time()
    self.cap = cv2.VideoCapture(self.rtsp)
    time_end_init_camera = time.time()

    # telegram
    self.telegrams = self.get_telegrams()    
    self.thread_telegram = None

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
      self.cap.release()
      text_status = f"Thời gian: {time.strftime(DATETIME_FORMAT)} \n Camera {self.rtsp} không hoạt động. Vui lòng kiểm tra lại"
      telegram_send_text_message(text_status, self.telegrams)

      send_email("thangxajk@gmail.com",f"SỰ CỐ CAMERA {self.rtsp}",text_status)
        
    self.frame_buffer = queue.Queue()
    self.telegram_buffer = queue.Queue()

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

def get_cameras_by_group_and_user(group_id, user_id):
  session = db_session()
  name = request.args.get("name", type=str, default='')

  cameras = session.query(
    Camera.id.label("id"),
    Camera.name.label("name"),
    Camera.rtsp.label("rtsp"),
    Camera.record_status.label("record_status"),
    CamerasUser.camera_status.label("status")
  ).join(
    CamerasUser, CamerasUser.camera_id == Camera.id
  ).filter(and_(
    Camera.group_id == group_id,
    CamerasUser.user_id == user_id,
    Camera.name.like(f"%{name}%")
  )).all()

  for i, camera in enumerate(cameras):
    features = session.query(Feature).join(
      FeatureAnalyze, Feature.id == FeatureAnalyze.feature_id
    ).filter(
      FeatureAnalyze.camera_id == camera.id
    ).all()

    cameras[i] = camera._asdict()
    cameras[i]['features_analyze'] = common_schema.dump(features)

  session.close()

  return cameras

def add_camera_service():
  try:
    name = request.get_json()['name'] 
    rtsp = request.get_json()['rtsp'] 
    record_status = request.get_json()['record_status'] 
    group_id = get_jwt_identity()['group_id'] 
    user_id = get_jwt_identity()['user_id'] 
    
    session = db_session()
    count = session.query(Camera).filter(and_(Camera.rtsp == rtsp, Camera.group_id == group_id)).count()
    
    # camera đã tồn tại
    if count >= 1:
      res = jsonify({"message": f"Cameras {name} đã tồn tại"}), 400

    else:
      # thêm camera và database
      camera = Camera(name, rtsp, record_status, group_id)
      session.add(camera)

      # lấy camera id
      camera_id = session.query(Camera).filter(and_(Camera.rtsp == rtsp, Camera.group_id == group_id)).first().id

      # cập nhật trạng thái camera cho mỗi user trong group
      users = session.query(User).filter(User.group_id == group_id).all()
      for user in users:
        camera_user = CamerasUser(camera_id, user.id, 1)
        session.add(camera_user)

      # tạo luồng chạy camera
      threads_camera_analyzer[camera.id] = CameraAnalyzer(camera)
      threads_camera_analyzer[camera.id].start()

      # bật khi hình
      if record_status == 1:
        threads_camera_analyzer[camera.id].set_record_status(True)

      session.commit()

      # lấy ra data mới
      cameras = get_cameras_by_group_and_user(group_id, user_id)

      res = jsonify({
        "message": f"Thêm cameras {name} thành công",
        "cameras": cameras_schema.dump(cameras)
      }), 200

    session.close()
    return res
  
  except KeyError:
    return jsonify({"message": "Bad request !"}), 400    

def get_cameras_service():
  try:
    group_id = get_jwt_identity()['group_id']
    user_id = get_jwt_identity()['user_id']
    
    cameras = get_cameras_by_group_and_user(group_id, user_id)
    return cameras_schema.dump(cameras), 200
  
  except KeyError:
    return jsonify({"message": "Bad request !"}), 400    

def start_record_camera_service(camera_id):
  try:
    user_id = get_jwt_identity()['user_id']
    group_id = get_jwt_identity()['group_id']

    session = db_session()
    camera = session.query(Camera).filter(Camera.id == camera_id).first()

    if camera is not None:
      # tạo luồng chạy camera
      if camera_id not in threads_camera_analyzer.keys():
        threads_camera_analyzer[camera.id] = CameraAnalyzer(camera)
        threads_camera_analyzer[camera.id].start()

      # bật ghi hình camera
      threads_camera_analyzer[camera_id].set_record_status(True)

      # lấy ra data mới
      cameras = get_cameras_by_group_and_user(group_id, user_id)
  
      res = jsonify({
        "message": f"Đã bật ghi hình camera {camera.name}",
        "cameras": cameras_schema.dump(cameras)
      }), 200
    
    else:
      res = jsonify({"message": f"Không tìm thấy camera"}), 404

    session.close()
    return res
    
  except KeyError:
    return jsonify({"message": "Bad request !"}), 400

def stop_recording_camera_service(camera_id):
  try:
    user_id = get_jwt_identity()['user_id']
    group_id = get_jwt_identity()['group_id']

    session = db_session()
    camera = session.query(Camera).filter(Camera.id == camera_id).first()
    
    if camera is not None and camera_id in threads_camera_analyzer.keys():
      threads_camera_analyzer[camera_id].set_record_status(False)
      
      # lấy ra data mới
      cameras = get_cameras_by_group_and_user(group_id, user_id)

      res = jsonify({
        "message": f"Đã tắt ghi hình camera {camera.name}",
        "cameras": cameras_schema.dump(cameras)
      }), 200

    else:
      res = jsonify({"message": f"Không tìm thấy camera !"}), 400
    
    session.close()
    return res
    
  except KeyError:
    return jsonify({"message": "Bad request !"}), 400    
