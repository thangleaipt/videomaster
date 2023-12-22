import os
from sqlalchemy import create_engine
# from dotenv import load_dotenv
import datetime
# load_dotenv()

# App config
# PORT = os.environ.get("PORT")
# HOST = os.environ.get("HOST")
# SECRET_KEY = os.environ.get("SECRET_KEY")

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATE_TIME_FORMAT = "%Y-%m-%d"
EXPIRES_DELTA_TOKEN = datetime.timedelta(days=1)

PROJECT_FOLDER = '../back-end'
STATIC_FOLDER = os.path.join(PROJECT_FOLDER, 'static')
REPORTS_IMAGES_FOLDER = os.path.join(STATIC_FOLDER, 'report_images')

# DB config
# DB_USER = os.environ.get('DB_USER')
# DB_PASS = os.environ.get('DB_PASS')
# DB_HOST = os.environ.get('DB_HOST')
# DB_NAME = os.environ.get('DB_NAME')




SECRET_KEY = "Wyj9GbtAHch1ibvlGdp52ZvNWZy1SZjFmZ"
PORT = 8005
HOST = "0.0.0.0"

# DB config
DB_USER = "face-recognition-app-local"
DB_PASS = "aipt2023"
DB_HOST = "localhost"
DB_NAME = "face-recognition-app-local"

db_engine = create_engine(f'mysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}')