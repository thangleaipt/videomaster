import json
from datetime import datetime

from server.config import DATETIME_FORMAT

class PersonModel:
    def __init__(self):
        self.list_face_name = []     # Name of person
        self.list_check_masks = []  
        self.list_check_glasses = []
        self.id = None      
        self.list_age = []
        self.list_gender = []
        self.id_db = None
        self.time = datetime.now().strftime(DATETIME_FORMAT)
        self.code_color = None
        self.name_color = None
        self.counting_tracking = 0
        self.counting_telegram = 0
        self.label_name = ""

        self.average_check_mask = None 
        self.average_check_glasses = None
        self.average_age = None  
        self.average_gender = None  
        self.color_clothes = []
        self.list_image_path = []
        self.face_image = None
        self.person_image = None
        self.is_front = 0
        self.list_feature = []
        self.score_face = 0
        self.real_time = 0


    def to_json(self):
        return {
            'list_face_name': self.list_face_name,
            'list_check_marks': self.list_check_marks,
            'id': self.id,
            'list_age': self.list_age,
            'list_gender': self.list_gender,
            'color_clothes': self.color_clothes,
            'list_image_path' : self.list_image_path
        }
    @staticmethod
    def from_json(json_data):
        return PersonModel(
            json_data['list_face_name'],
            json_data['list_check_marks'],
            json_data['id'],
            json_data['list_age'],
            json_data['list_gender'],
            json_data['color_clothes'],
            json_data['list_image_path']
        )