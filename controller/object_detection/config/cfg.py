
from easydict import EasyDict as edict


__C                           = edict()
# Consumers can get config by: from config import cfg

cfg                           = __C

# YOLO options
__C.YOLO                      = edict()
__C.YOLO                      = edict()

__C.YOLO.CLASSES              = './controllers/object_detection/classes.names'