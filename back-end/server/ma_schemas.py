from .extension import ma

class UsersSchema(ma.Schema):
  class Meta:
    fields = ("name", "email", "phone", "role_name")

class VideosSchema(ma.Schema):
  class Meta:
    fields = ("id", "path", "time")

class ReportsSchema(ma.Schema):
  class Meta:
    fields = ("id", "person_name", "age", "gender", "mask", "code_color", "time", "images")

class ImagesSchema(ma.Schema):
  class Meta:
    fields = ("id", "path")