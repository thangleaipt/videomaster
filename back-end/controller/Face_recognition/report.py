
from datetime import datetime

from server.config import DATETIME_FORMAT

from server.reports.services import add_report_service
import uuid


def send_report_to_db(camera_analyze):
    for i, person_model in enumerate(camera_analyze.list_person_model):
        if len(person_model.list_image_path) == []:
            continue
        if person_model.average_gender is not None:
            if person_model.average_gender[0] == "male" or person_model.average_gender == "male":
                gender = 1
            elif person_model.average_gender[0] == "female" or person_model.average_gender == "female":
                gender = 0
        else:
            gender = 2
        # gender = person_model.average_gender
        if person_model.average_age is not None:
            age = int(person_model.average_age)
        else: 
            age = 0
        person_model_time = datetime.strptime(person_model.time, "%Y-%m-%d %H:%M:%S")
        # Lấy timestamp
        time_model = person_model_time.timestamp()
        # time_model = datetime.timestamp(person_model.time)
        if person_model.average_check_mask == "Mask":
            mask = 1
        else:
            mask = 0
        if person_model.average_check_glasses == True:
            glasses = 1
        else:
            glasses = 0
        main_color_clothes = person_model.code_color
        name_color = person_model.name_color
        if person_model.label_name is None or person_model.label_name == "":
            person_name = f"random_{uuid.uuid4()}"
        else:
            person_name = person_model.label_name
        print(f"Report: {person_name}: {person_model.id}, {age}, {gender}, {mask}, {main_color_clothes}, {time_model}, {person_model.list_image_path}")
        add_report_service(camera_analyze.video_path_report, person_name, age, gender, mask, main_color_clothes, time_model, person_model.list_image_path)