from flask import jsonify, request
from flask_jwt_extended import create_access_token, get_jwt_identity
from server.config import EXPIRES_DELTA_TOKEN
from server.extension import db_session
from server.models import User, Role
from sqlalchemy import or_
from passlib.hash import bcrypt
from server.ma_schemas import UsersSchema

user_schema = UsersSchema()

def user_login_service():
  try:
    username = request.get_json()["username"]
    password = request.get_json()["password"]
    
    session = db_session()

    # kiểm tra thông tin đăng nhập
    user = session.query(User).filter(or_(
      User.email == username, 
      User.phone == username
    )).first()

    if user is not None and bcrypt.verify(password, user.password):
      token = create_access_token(
        identity = {"user_id": user.id}, 
        expires_delta = EXPIRES_DELTA_TOKEN
      )
      res = jsonify({"message": "Đăng nhập thành công", "token": token }), 200

    else:
      res = jsonify({"message": "Email hoặc số điện thoại không đúng !. Vui lòng kiểm tra lại"}), 401
    
    session.close()
    return res
    
  except KeyError:
    return jsonify({"message": "Bad request !"}), 400

def get_user_info_by_token_service():
  user_id = get_jwt_identity()['user_id']
  session = db_session()
  
  user = session.query(
    User.name.label('name'), 
    User.email.label('email'), 
    User.phone.label('phone'), 
    Role.name.label('role_name')
  ).join(
    Role, User.role_id == Role.id
  ).filter(
    User.id == user_id
  ).first()

  if user is not None:
    res = user_schema.dump(user), 200
    
  else:
    res = jsonify({"message": "Không tìm thấy user !"}), 401

  session.close()
  return res