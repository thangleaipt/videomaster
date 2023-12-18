from server.extension import db
from sqlalchemy.dialects.mssql import INTEGER, NVARCHAR, VARCHAR, DATETIME
from sqlalchemy import Column, ForeignKey

class Role(db.Model):
  __tablename__ = "Roles"
  id = Column(INTEGER, primary_key=True, autoincrement=True)
  name = Column(NVARCHAR(50), nullable=False)

  def __init__(self, name):
    self.name = name

class User(db.Model):
  __tablename__ = "Users"
  id = Column(INTEGER, primary_key=True, autoincrement=True)
  name = Column(NVARCHAR(50), nullable=False)
  email = Column(VARCHAR(50), nullable=False, unique=True)
  phone = Column(VARCHAR(15), nullable=False, unique=True)
  password  = Column(VARCHAR(150), nullable=False)
  role_id = Column(INTEGER, ForeignKey(Role.id), nullable=False)

  def __init__(self, name, email, phone, password):
    self.name = name
    self.email = email
    self.phone = phone
    self.password = password

class Camera(db.Model):
  __tablename__ = "Cameras"
  id = Column(INTEGER, primary_key=True, autoincrement=True) 
  name = Column(NVARCHAR(100), nullable=False)
  rtsp = Column(VARCHAR(100), nullable=False)
  record_status = Column(INTEGER, nullable=False)
  group_id = Column(INTEGER, ForeignKey(Role.id), nullable=False)

  def __init__(self, name, rtsp, record_status, group_id):
    self.name = name
    self.rtsp = rtsp
    self.record_status = record_status
    self.group_id = group_id


class Record(db.Model):
  __tablename__ = "Records"
  id = Column(INTEGER, primary_key=True, autoincrement=True) 
  start = Column(DATETIME, nullable=False)
  end = Column(DATETIME, nullable=True)
  path = Column(VARCHAR(255), nullable=False)
  camera_id = Column(INTEGER, ForeignKey(Camera.id), nullable=False)
  update_by = Column(INTEGER, ForeignKey(Role.id), nullable=True)

  def __init__(self, start, end, path, camera_id, update_by):
    self.start = start
    self.end = end
    self.path = path
    self.camera_id = camera_id
    self.update_by = update_by

class Video(db.Model):
  __tablename__ = "Videos"

  id = Column(INTEGER, primary_key=True, autoincrement=True)
  path = Column(NVARCHAR(255), nullable=False)
  time = Column(INTEGER, nullable=False)

  def __init__(self, path, time):
    self.path = path
    self.time = time

class Report(db.Model):
  __tablename__ = "Reports"

  id = Column(INTEGER, primary_key=True, autoincrement=True)
  person_name = Column(NVARCHAR(50), nullable=True)
  age = Column(INTEGER, nullable=True)
  gender = Column(INTEGER, nullable=True)
  mask = Column(INTEGER, nullable=True)
  code_color = Column(VARCHAR(20), nullable=True)
  time = Column(INTEGER, nullable=False)
  video_id = Column(INTEGER, ForeignKey(Video.id), nullable=False)

  def __init__(self, person_name, age, gender, mask, code_color, time, video_id):
    self.person_name = person_name
    self.age = age
    self.gender = gender
    self.mask = mask
    self.code_color = code_color
    self.time = time
    self.video_id = video_id

class ReportImage(db.Model):
  __tablename__ = "ReportImages"

  id = Column(INTEGER, primary_key=True, autoincrement=True)
  path = Column(NVARCHAR(255), nullable=False)
  report_id = Column(INTEGER, ForeignKey(Report.id), nullable=False)

  def __init__(self, path, report_id):
    self.path = path
    self.report_id = report_id