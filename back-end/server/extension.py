from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import sessionmaker
from flask_marshmallow import Marshmallow
from .config import db_engine

db = SQLAlchemy()
db_session = sessionmaker(bind = db_engine)
ma = Marshmallow()