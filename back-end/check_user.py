import requests
import subprocess
from requests.exceptions import ConnectionError
from server.extension import db_session
from server.models import User
from sqlalchemy import or_
from passlib.hash import bcrypt

# Tạo db và cấp quyền vào db
# create user `face-recognition-app-local`@'%' identified by 'aipt2023';
# create database `face-recognition-app-local`;
# grant all on `face-recognition-app-local`.* to `face-recognition-app-local`@'%';

# tạo user admin
# insert into roles(name) values('ADMIN');
# insert into users(name, email, phone, password, role_id) values ('admin', 'admin@gmail.com', '0326660728', '$2b$12$QGb1Lbi./rlXQuN9Dohxo.whUJUfzekCZiaXiIaDMYfa2luXYxr1K', 1);
# commit;

def verify_authorization(username, password):
  session = db_session()

  try:
    machine_id = subprocess.check_output('wmic csproduct get uuid').split()[1].decode('utf-8')

    # call API
    response = requests.get('http://117.4.254.94:8009/app-authorization', headers={
      "key": "ozthrmnDNKMpHxlwtEfVTaKJiFGlPsdmfPBROLCxTfwgYwMoCi",
      "app-code": "APP001",
      "device": machine_id
    })

    message = ""
    if "message"in response.json().keys():
      message = response.json()["message"]

    verify_status = False
    
    if response.status_code == 200:
      user = session.query(User).filter(or_(
        User.email == username, 
        User.phone == username
      )).first()

      if user is not None and bcrypt.verify(password, user.password):
        verify_status = True
        message = "Đăng nhập thành công"

      else:
        message = "Sai thông tin đăng nhập !"
    
    return {
      "message": message,
      "response_status_code": response.status_code,
      "verify_status": verify_status
    }
  
  except ConnectionError as e:
    return {
      "message": f"Lỗi kết nối với API xác thực app",
      "response_status_code": 500,
      "verify_status": False,
      "error": f"{e}",
    }

  except Exception as e:
    return {
      "message": f"{e}",
      "response_status_code": 500,
      "verify_status": False
    }
  
  finally:
    session.close()