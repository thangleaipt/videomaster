from flask import Blueprint
from .services import *
from flask_jwt_extended import jwt_required

reports = Blueprint('reports', __name__)

@reports.route('/get-videos', methods=["GET"])
@jwt_required()
def get_videos():
  return get_videos_service()

@reports.route('/get-reports/<int:video_id>', methods=["GET"])
@jwt_required()
def get_reports(video_id):
  return get_reports_service(video_id)