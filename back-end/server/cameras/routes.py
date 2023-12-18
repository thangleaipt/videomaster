from flask import Blueprint
from flask_jwt_extended import jwt_required
from .service import *

cameras = Blueprint('cameras', __name__)

@cameras.route('/add-camera', methods = ["POST"])
@jwt_required()
def add_camera():
  return add_camera_service()

@cameras.route('/get-cameras', methods = ["GET"])
@jwt_required()
def get_cameras():
  return get_cameras_service()

@cameras.route('/start-record-camera/<int:camera_id>', methods = ["POST"])
@jwt_required()
def start_record_camera(camera_id):
  return start_record_camera_service(camera_id)

@cameras.route('/stop-recording-camera/<int:camera_id>', methods = ["POST"])
@jwt_required()
def stop_recording_camera(camera_id):
  return stop_recording_camera_service(camera_id)

@cameras.route('/show-camera/<int:camera_id>', methods = ["GET"])
def show_camera(camera_id):
  return show_camera_service(camera_id)

@cameras.route("/update-camera/<int:camera_id>", methods=["PUT"])
@jwt_required()
def update_camera(camera_id):
  return update_camera_service(camera_id)

@cameras.route("/re-start-camera/<int:camera_id>", methods=["POST"])
@jwt_required()
def re_start_camera(camera_id):
  return re_start_camera_service(camera_id)