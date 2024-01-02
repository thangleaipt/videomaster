from deepface import DeepFace
from deepface.commons import functions, realtime, distance as dst
import numpy as np
import time
import cv2
from os import path
import pickle
import os
from tqdm import tqdm
import pandas as pd

from model.logger import miolog

os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;udp'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class FaceAnalysis:
    def __init__(self, db_path="dataset", model_name="VGG-Face", detector_backend="opencv", distance_metric="cosine", enable_face_analysis=True, source=0, time_threshold=5, frame_threshold=5, *args, **kwargs):
        self.db_path = db_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.distance_metric = distance_metric
        self.enable_face_analysis = enable_face_analysis
        self.source = source
        self.time_threshold = time_threshold
        self.frame_threshold = frame_threshold
        self.target_size = functions.find_target_size(model_name=model_name)
        self.enforce_detection = False
        self.align = True
        self.normalization = "base"
        self.silent = True
        self.list_labels = []   
        self.model = DeepFace.build_model(model_name)
        self.representations = self.load_db()


    def represent(
        self,
        img,
        region,
        confidence
    ):
        model = self.model
        normalization = self.normalization
        # custom normalization
        img = functions.normalize_input(img=img, normalization=normalization)

        # represent
        if "keras" in str(type(model)):
            # new tf versions show progress bar and it is annoying
            embedding = model.predict(img, verbose=0)[0].tolist()
        else:
            # SFace and Dlib are not keras models and no verbose arguments
            embedding = model.predict(img)[0].tolist()

        resp_obj = {}
        resp_obj["embedding"] = embedding
        resp_obj["facial_area"] = region
        resp_obj["face_confidence"] = confidence

        return resp_obj
    
    def delete(self, label):
        try:
            if label not in self.list_labels:
                print(f'{label} does not exist in dataset')
                return
            indices = [i for i, x in enumerate(self.list_labels) if x == label]
            # for idx in indices:
            image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
            image_file_found = False
            image_file_path = None
            for extension in image_extensions:
                image_file_path = os.path.join(self.deep_face.db_path, f'{label}.{extension}')
                if os.path.exists(image_file_path):
                    image_file_found = True
                    miolog.info(f"[get_label_images]: Image file found: {image_file_path}")
                    break
            if image_file_found and image_file_path is not None:
                path_label = image_file_path

                if path.exists(path_label):
                    os.remove(path_label)
                    miolog.info(f'[analyze_video][delete]: delete {label} from dataset')
                    
                if type(self.list_labels) == list:
                    for idx in sorted(indices, reverse=True):
                        del self.labels[idx]
                else:
                    self.list_labels = np.delete(self.list_labels, indices)
                miolog.info(f'labels: {len(self.list_labels)}')
                # remove label from representations
                if type(self.representations) == list:
                    for idx in sorted(indices, reverse=True):
                        del self.representations[idx]
                else:
                    self.representations = np.delete(self.representations, indices)
                miolog.info(f'representations: {len(self.representations)}')
                with open(f"{self.db_path}/representations_{self.model_name}.pkl", "wb") as f:
                    pickle.dump(self.representations, f)
        except Exception as e:
            miolog.exception('delete label fail', e)

    
    def train(self, image, label):
        try:
            db_path = self.db_path
            model_name = self.model_name

            file_name = f"representations_{model_name}.pkl"
            file_name = file_name.replace("-", "_").lower()
            if path.exists(db_path + "/" + file_name):
                print(
                    f"WARNING: Representations for images in {db_path} folder were previously stored"
                    + f" in {file_name}. If you added new instances after the creation, then please "
                    + "delete this file and call find function again. It will create it again."
                )

                with open(f"{db_path}/{file_name}", "rb") as f:
                    representations = pickle.load(f)
            
                img_objs = functions.extract_faces(
                        img=image,
                        target_size=self.target_size,
                        detector_backend=self.detector_backend,
                        grayscale=False,
                        enforce_detection=self.enforce_detection,
                        align=self.align,
                    )
                for img, region, confidence in img_objs:
                    embedding_obj = self.represent(img, region, confidence)

                    img_representation = embedding_obj["embedding"]

                    instance = []
                    instance.append(label)
                    instance.append(img_representation)
                    representations.append(instance)

                # check path image
                image_extensions = ['jpg', 'jpeg', 'png', 'gif', 'bmp']
                image_file_found = False
                image_file_path = None
                for extension in image_extensions:
                    image_file_path = os.path.join(self.deep_face.db_path, f'{label}.{extension}')
                    if os.path.exists(image_file_path):
                        image_file_found = True
                        miolog.info(f"[get_label_images]: Image file found: {image_file_path}")
                        break
                if image_file_found and image_file_path is not None:
                    # delete image
                    os.remove(image_file_path)
                cv2.imwrite(f"{db_path}/{label}.jpg", image)
                with open(f"{db_path}/{file_name}", "wb") as f:
                    pickle.dump(representations, f)
            
            else:
                self.load_db()
                
        except Exception as e:
            representations = []
    
    def is_path_inside_db(self,db_path, path_input):
        db_path = os.path.abspath(db_path)
        path_input = os.path.abspath(path_input)
        common_path = os.path.commonpath([db_path, path_input])
        if common_path == db_path:
            return True
        else:
            return False 

    def load_db(self):
        db_path = self.db_path
        model_name = self.model_name
        silent = self.silent
        target_size = self.target_size
        detector_backend = self.detector_backend
        enforce_detection = self.enforce_detection
        align = self.align
        normalization = self.normalization

        

        file_name = f"representations_{model_name}.pkl"
        file_name = file_name.replace("-", "_").lower()

        if path.exists(db_path + "/" + file_name):

            if not silent:
                print(
                    f"WARNING: Representations for images in {db_path} folder were previously stored"
                    + f" in {file_name}. If you added new instances after the creation, then please "
                    + "delete this file and call find function again. It will create it again."
                )
            list_train_path = []
            list_sub_train_path = []
            list_trained_path = []
            with open(f"{db_path}/{file_name}", "rb") as f:
                representations = pickle.load(f)
                for rep in representations:
                    list_trained_path.append(rep[0])
                    name = os.path.basename(rep[0])
                    label = os.path.splitext(name)[0]
                    self.list_labels.append(label)

                # for r, _, sub_files in os.walk(db_path):
                #     for sub_file in sub_files:
                #         sub_path = os.path.join(db_path, sub_file)
                #         if os.path.isfile(sub_path) and sub_path.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                #             if sub_path not in list_trained_path:
                #                 list_train_path.append(sub_path)
                    
                for item in os.listdir(db_path):
                    item_path = os.path.join(db_path, item)
                    if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                        if item_path not in list_trained_path:
                            list_train_path.append(item_path)

                    elif os.path.isdir(item_path):
                        for _, _, sub_files in os.walk(item_path):
                            for sub_file in sub_files:
                                if (
                                    sub_file.lower().endswith((".jpg", ".jpeg", ".png"))
                                ):
                                    sub_file_path = os.path.join(item_path, sub_file)
                                    if sub_file_path not in list_trained_path:
                                        list_sub_train_path.append(sub_file_path)
 

                pbar = tqdm(
                range(0, len(list_train_path)),
                desc="Finding representations",
                disable=silent,
            )
                sub_pbar = tqdm(
                range(0, len(list_sub_train_path)),
                desc="Finding representations",
                disable=silent,
            )
                
                for index in pbar:
                    employee = list_train_path[index]

                    img_objs = functions.extract_faces(
                        img=employee,
                        target_size=target_size,
                        detector_backend=detector_backend,
                        grayscale=False,
                        enforce_detection=enforce_detection,
                        align=align,
                    )

                    for img, region, confidence in img_objs:
                        embedding_obj = self.represent(img, region, confidence)

                        img_representation = embedding_obj["embedding"]

                        instance = []
                        # Get name of image
                        name = os.path.basename(employee)
                        label = os.path.splitext(name)[0]

                        instance.append(employee)
                        instance.append(img_representation)
                        instance.append(label)
                        representations.append(instance)
                        self.list_labels.append(label)

                for index in sub_pbar:
                    employee = list_sub_train_path[index]

                    img_objs = functions.extract_faces(
                        img=employee,
                        target_size=target_size,
                        detector_backend=detector_backend,
                        grayscale=False,
                        enforce_detection=enforce_detection,
                        align=align,
                    )

                    for img, region, confidence in img_objs:
                        embedding_obj = self.represent(img, region, confidence)

                        img_representation = embedding_obj["embedding"]

                        instance = []
                        # Get name of image
                        path_folder = os.path.dirname(employee)
                        label = os.path.basename(path_folder)

                        instance.append(employee)
                        instance.append(img_representation)
                        instance.append(label)
                        representations.append(instance)
                        self.list_labels.append(label)

            with open(f"{db_path}/{file_name}", "wb") as f:
                pickle.dump(representations, f)          

            if not silent:
                print("There are ", len(representations), " representations found in ", file_name)

        else:  # create representation.pkl from scratch
            employees = []
            sub_employees = []

            for item in os.listdir(db_path):
                item_path = os.path.join(db_path, item)
                if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                    employees.append(item_path)

                elif os.path.isdir(item_path):
                    for _, _, sub_files in os.walk(item_path):
                        for sub_file in sub_files:
                            if (
                                sub_file.lower().endswith((".jpg", ".jpeg", ".png"))
                            ):
                                sub_file_path = os.path.join(item_path, sub_file)
                                sub_employees.append(sub_file_path)

            if len(employees) == 0:
                raise ValueError(
                    "There is no image in ",
                    db_path,
                    " folder! Validate .jpg or .png files exist in this path.",
                )

            # ------------------------
            # find representations for db images
            representations = []

            # for employee in employees:
            pbar = tqdm(
                range(0, len(employees)),
                desc="Finding representations",
                disable=silent,
            )
            sub_pbar = tqdm(
                range(0, len(sub_employees)),
                desc="Finding representations",
                disable=silent,
            )
            for index in pbar:
                employee = employees[index]

                img_objs = functions.extract_faces(
                    img=employee,
                    target_size=target_size,
                    detector_backend=detector_backend,
                    grayscale=False,
                    enforce_detection=enforce_detection,
                    align=align,
                )

                for img, region, confidence in img_objs:
                    embedding_obj = self.represent(img, region, confidence)

                    img_representation = embedding_obj["embedding"]

                    instance = []
                    # Get name of image
                    name = os.path.basename(employee)
                    label = os.path.splitext(name)[0]

                    instance.append(employee)
                    instance.append(img_representation)
                    instance.append(label)
                    representations.append(instance)
                    self.list_labels.append(label)

            for index in sub_pbar:
                sub_employee = sub_employees[index]

                sub_img_objs = functions.extract_faces(
                    img=sub_employee,
                    target_size=target_size,
                    detector_backend=detector_backend,
                    grayscale=False,
                    enforce_detection=enforce_detection,
                    align=align,
                )

                for img, region, confidence in sub_img_objs:
                    embedding_obj = self.represent(img, region, confidence)

                    img_representation = embedding_obj["embedding"]

                    instance = []
                    # Get name of image
                    path_folder = os.path.dirname(sub_employee)
                    label = os.path.basename(path_folder)

                    instance.append(sub_employee)
                    instance.append(img_representation)
                    instance.append(label)
                    representations.append(instance)
                    self.list_labels.append(label)

            # -------------------------------
            with open(f"{db_path}/{file_name}", "wb") as f:
                pickle.dump(representations, f)

            if not silent:
                print(
                    f"Representations stored in {db_path}/{file_name} file."
                    + "Please delete this file when you add new identities in your database."
                )
        return representations

    def recognition(self, target_img, target_region, target_confidence):
        representations = self.representations
        model_name = self.model_name
        distance_metric = self.distance_metric
        silent = self.silent

        # now, we got representations for facial database
        df = pd.DataFrame(representations, columns=["identity", f"{model_name}_representation", "labels"])

        # img path might have more than once face
        target_embedding_obj = self.represent(
            img=target_img,
            region=target_region,
            confidence=target_confidence
        )

        target_representation = target_embedding_obj["embedding"]

        result_df = df.copy()  # df will be filtered in each img
        result_df["source_x"] = target_region["x"]
        result_df["source_y"] = target_region["y"]
        result_df["source_w"] = target_region["w"]
        result_df["source_h"] = target_region["h"]

        distances = []
        for index, instance in df.iterrows():
            source_representation = instance[f"{model_name}_representation"]

            if distance_metric == "cosine":
                distance = dst.findCosineDistance(source_representation, target_representation)
            elif distance_metric == "euclidean":
                distance = dst.findEuclideanDistance(source_representation, target_representation)
            elif distance_metric == "euclidean_l2":
                distance = dst.findEuclideanDistance(
                    dst.l2_normalize(source_representation),
                    dst.l2_normalize(target_representation),
                )
            else:
                raise ValueError(f"invalid distance metric passes - {distance_metric}")

            distances.append(distance)

            # ---------------------------

        result_df[f"{model_name}_{distance_metric}"] = distances
        

        threshold = dst.findThreshold(model_name, distance_metric)
        
        result_df = result_df.drop(columns=[f"{model_name}_representation"])
        result_df = result_df[result_df[f"{model_name}_{distance_metric}"] <= threshold]
        result_df = result_df.sort_values(
            by=[f"{model_name}_{distance_metric}"], ascending=True
        ).reset_index(drop=True)
        # -----------------------------------
        return result_df

    def analysis_realtime(self):
        model_name = self.model_name
        source = self.source
        # global variables
        text_color = (0, 255, 255)

        # find custom values for this input set
        target_size = functions.find_target_size(model_name=model_name)
        # ------------------------
        DeepFace.build_model(model_name=model_name)
        print(f"facial recognition model {model_name} is just built")

        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)  # webcam
        # Check if the camera opened successfully
        if not cap.isOpened():
            print("Error: Could not open camera.")
            exit()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out_imwrite = cv2.VideoWriter(f"output_{self.source}.avi", fourcc, 30, (width, height))

        index = 0

        while True:
            ret, frame = cap.read()
            index += 1
            if not ret:
                print("Error: Could not read frame. Reopening the camera...")
                cap.release()  # Release the camera
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)  # Reopen the camera
                if not cap.isOpened():
                    print("Error: Could not reopen camera.")
                    break
                continue
            img = frame.copy()
            # just extract the regions to highlight in webcam
            time_start_detect_face = time.time()
            face_objs = functions.extract_faces(
                img=img,
                target_size=target_size,
                detector_backend=self.detector_backend,
                grayscale=False,
                enforce_detection=self.enforce_detection,
                align=self.align,
            )
            time_end_detect_face = time.time()
            print(f"time detect face: {time_end_detect_face - time_start_detect_face}")
            # for face_obj in face_objs:
            if len(face_objs) > 0:
                for custom_img, region, confidence in face_objs:
                    x, y, w, h = int(region["x"]), int(region["y"]), int(region["w"]), int(region["h"])
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw rectangle to main image
                
                    dfs = self.recognition(
                        target_img = custom_img,
                        target_region = region,
                        target_confidence = confidence
                    )

                    if dfs.shape[0] > 0:
                        candidate = dfs.iloc[0]
                        label = candidate["identity"]
                        label = label.split("/")[-1]
                    else:
                        label = "Unknown"

                    cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
            # resize 1280, 720
            out_imwrite.write(img)
            img = cv2.resize(img, (640, 480))
            cv2.imshow(f"Camera_{self.source}", img)
            
            if cv2.waitKey(1) & 0xFF == ord("q"):  # press q to quit
                print("Break")
                break
            
        cap.release()
        cv2.destroyAllWindows()

    def analysis_frame(self, frame):
        # global variables
        text_color = (0, 255, 255)
        # find custom values for this input set
        img = frame.copy()
        # just extract the regions to highlight in webcam
        time_start_detect_face = time.time()
        face_objs = functions.extract_faces(
            img=img,
            target_size=self.target_size,
            detector_backend=self.detector_backend,
            grayscale=False,
            enforce_detection=self.enforce_detection,
            align=self.align,
        )
        time_end_detect_face = time.time()
        print(f"time detect face: {time_end_detect_face - time_start_detect_face}")
        # for face_obj in face_objs:
        if len(face_objs) > 0:
            for custom_img, region, confidence in face_objs:
                x, y, w, h = int(region["x"]), int(region["y"]), int(region["w"]), int(region["h"])
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # draw rectangle to main image
            
                dfs = self.recognition(
                    target_img = custom_img,
                    target_region = region,
                    target_confidence = confidence
                )

                if dfs.shape[0] > 0:
                    candidate = dfs.iloc[0]
                    label = candidate["labels"]
                    # label = label.split("/")[-1]
                else:
                    label = "Unknown"

                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2)
    
        # img = cv2.resize(img, (1280, 720))
        return img




