import cv2
import time
import numpy as np

from utils import iou_cal, check_iou, check_same_box

CONFIDENCE_THRESHOLD = 0.8
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

yolov5_model = cv2.dnn.readNetFromONNX("./yolov5.onnx")

print("Loaded model")


def detech_frame_v5(frame, model):
    list_box = []
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (640, 640), swapRB=True)
    model.setInput(blob)
    w, h = frame.shape[1], frame.shape[0]
    w /= 640
    h /= 640
    img = model.forward()
    idx = -1
    for val in img[0]:
        val[..., 0] *= w
        val[..., 1] *= h
        val[..., 2] *= w
        val[..., 3] *= h
        idx += 1
        xmin, ymin, xmax, ymax, confidence, class_0, class_1 = val

        if confidence > CONFIDENCE_THRESHOLD:
            list_box.append(val)

    list_box = check_iou(list_box)
    for box in list_box:
        xmin, ymin, xmax, ymax, confidence, class_0, class_1 = box
        classid = 0 if class_0 > class_1 else 1
        box = int(xmin - 10), int(ymin - 10), int(xmax), int(ymax - 10)
        color = COLORS[int(classid) % len(COLORS)]
        label = "{} {}".format(class_names[classid], confidence)
        cv2.rectangle(frame, box, color, 1)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    # print(len(list_box))

    return frame

