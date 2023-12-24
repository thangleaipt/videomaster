from flask import Flask
from .extension import db
from .models import *
from .config import db_engine, PROJECT_FOLDER, STATIC_FOLDER

from .users.routes import users
from .reports.routes import reports

def create_db():
  # app = Flask(__name__, root_path=PROJECT_FOLDER, static_folder=STATIC_FOLDER)
  # app.config.from_object(config)

  # with app.app_context():
    db.metadata.create_all(db_engine)

  # register blueprint
  # app.register_blueprint(users)
  # app.register_blueprint(reports)

  # return app