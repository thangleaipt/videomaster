from flask import Blueprint
from .services import *
from flask_jwt_extended import jwt_required

users = Blueprint('users', __name__)

@users.route('/user/login', methods=["POST"])
def user_login():
    return user_login_service()

@users.route('/get-user-info-by-token', methods=["GET"])
@jwt_required()
def get_user_info_by_token():
   return get_user_info_by_token_service()
