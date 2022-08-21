import cv2
import time
import imutils
import os

import numpy as np


class PeopleDetectorV7:

    CONFIDENCE_THRESHOLD = 0.5 #0.35
    NMS_THRESHOLD = 0.4 #0.4
    COLORS = [(0, 255, 255), (255, 255, 0),  (0, 255, 0), (255, 0, 0)]
    class_names = []
    model = None

    def __init__(self):
        pth = os.path.dirname(__file__)

        # switch to Yolov7 tiny, the latest yolo model, super fast
        class_file_name = os.path.join(pth, "yolov7/coco.names")
        weight_file_name = os.path.join(pth, "yolov7/yolov7-tiny.cfg")
        cfg_file_name = os.path.join(pth, "yolov7/yolov7-tiny.weights")


        with open(class_file_name, "r") as f:
            self.class_names = [cname.strip() for cname in f.readlines()]

        net = cv2.dnn.readNet(weight_file_name, cfg_file_name)
        # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

        #self.face_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def update(self, image):


        print("update")
        # 显示图片
        #cv2.putText(frame, "{}people".format(num), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (142, 125, 52), 1)

    def detect(self, image):

        print("detect people")




