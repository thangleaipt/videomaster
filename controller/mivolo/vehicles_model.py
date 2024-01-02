import json

class VehicleModel:
    def __init__(self):
        self.class_name = None     # Name of person
        self.color = None     
        self.id = None      
        self.image_path = []

    def to_json(self):
        return {
            'class_name': self.class_name,
            'color': self.color,
            'id': self.id,
            'image_path' : self.image_path
        }
    @staticmethod
    def from_json(json_data):
        return VehicleModel(
            json_data['class_name'],
            json_data['color'],
            json_data['id'],
            json_data['image_path']
        )