from server.config import SECRET_KEY
import jwt
    
def verify_authorization(token):
  try:
    token = f"{token}".replace('Bearer ', '')
    data = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
    
    if data is not None:
      return True
    else:
      return False
  except Exception:
    return False
  
def get_file_extension(filename):
  return f"{filename}".split(".")[::-1][0]

  
