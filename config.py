
WEIGHT_FOLDER = "weights"
# WEIGHT_FOLDER = r"back-end\weights"
DATASET_PEOPLE_FOLDER = "dataset"
# DATASET_PEOPLE_FOLDER = r"back-end\dataset"
DEVICE = "cuda"
# STATIC_FOLDER = r"back-end\static"
STATIC_FOLDER = "static"

import os
documents = documents_path = os.path.join(os.path.expanduser("~"), "Documents")
STATIC_FOLDER = os.path.join(documents, "VIDEOMASTER AI")
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)
DATASET_PEOPLE_FOLDER = os.path.join(STATIC_FOLDER, "DATASET")
if not os.path.exists(DATASET_PEOPLE_FOLDER):
    os.makedirs(DATASET_PEOPLE_FOLDER)