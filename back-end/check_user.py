import requests
import subprocess
from requests.exceptions import ConnectionError


def verify_authorization(username, password):
  try:
    machine_id = subprocess.check_output('wmic csproduct get uuid').split()[1].decode('utf-8')

    # call API
    response = requests.get('http://117.4.254.94:5005/app-authorization', headers={
      "key": "ozthrmnDNKMpHxlwtEfVTaKJiFGlPsdmfPBROLCxTfwgYwMoCi",
      "app-code": "APP001",
      "device": machine_id
    })

    message = ""
    if "message"in response.json().keys():
      message = response.json()["message"]

    verify_status = False
    
    if response.status_code == 200:
      if username == "0326660728" and password == "123456":
        verify_status = True
      
      else:
        message = "Sai thông tin đăng nhập !"
    
    return verify_status
     
  
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

print(verify_authorization(1,1))