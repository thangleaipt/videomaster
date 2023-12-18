
import os
import sys

sys.path.append(os.getcwd())
import numpy as np
import torch


def load_model_yolov5(path_hub, path_model):
    return torch.hub.load(repo_or_dir=path_hub, model='custom', path=path_model, source="local", force_reload=False)


def predict_object_detector_yolov5(model, image):
    model.iou = 0.4  # [0.99,0.1]
    model.conf = 0.5
    model.multi_label = True

    results = model(image)
    array_results = results.pandas().xyxy[0].values
    if len(array_results) > 0:
        array_results[:, 2] = array_results[:, 2] - array_results[:, 0]
        array_results[:, 3] = array_results[:, 3] - array_results[:, 1]
        object = array_results[:, 0:6]
        object = np.vstack(object).astype(np.float64)
        return (object[(np.where(object[:, 5] == 1))], object[(np.where(object[:, 5] == 0))], None)
    else:
        return None

