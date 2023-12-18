import cv2
import time
import glob
import numpy as np

CONFIDENCE_THRESHOLD = [0.5, 0.7, 0.7]
NMS_THRESHOLD = [0.4, 0.55, 0.55]
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

# initialize our lists of detected bounding boxes, confidences, and
# class IDs, respectively
__NUM_CLASSES = 3

class_names = []

with open("/Users/v-miodohien/Desktop/Opencv_Yolov4Yolovv/model/yolov4_coco.names", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

weightsPath = "./model/yolov4-crowdhuman-416x416_umbrella_20211208.weights"
configPath = "./model/yolov4-crowdhuman-416x416.cfg"

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

for filename in glob.glob('/Users/v-miodohien/Downloads/Tara_Mio_Prepared_Images_20211224_720p/*.jpg'):
    # filename = '/Users/v-miodohien/Downloads/Tara_Mio_Prepared_Images_20211224_720p/01_73371852.jpg_resized.jpg'

    boxes = [[] for _ in range(__NUM_CLASSES)]
    confidences = [[] for _ in range(__NUM_CLASSES)]
    classIDs = [[] for _ in range(__NUM_CLASSES)]

    print(filename)
    image = cv2.imread(filename)
    (H, W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if class_names[classID] == "face":
                continue

            if confidence > 0.4:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes[classID].append([x, y, int(width), int(height)])
                confidences[classID].append(float(confidence))
                classIDs[classID].append(classID)
    # print(boxes)
    for c in range(__NUM_CLASSES):
        idxs = cv2.dnn.NMSBoxes(boxes[c], confidences[c], CONFIDENCE_THRESHOLD[c], NMS_THRESHOLD[c])
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[c][i][0], boxes[c][i][1])
                (w, h) = (boxes[c][i][2], boxes[c][i][3])
                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in COLORS[classIDs[c][i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(class_names[classIDs[c][i]], confidences[c][i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)
    # show the output image
    cv2.imshow("Image", image)
    cv2.waitKey(0)