## Tiny Yoyov7 model  --  速度相当可以
import cv2
import time
import os

### check performance gap
import imutils
from imutils.video import VideoStream

from multiprocessing import Process
import threading


def run_stream():
    print("thread 1")

    CONFIDENCE_THRESHOLD = 0.35
    NMS_THRESHOLD = 0.4
    COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

    pth = os.path.dirname(__file__)

    # switch to Yolov7 tiny, the latest yolo model, super fast
    class_file_name = os.path.join(pth, "yolov7/coco.names")
    cfg_file_name = os.path.join(pth, "yolov7/yolov7-tiny.cfg")
    weight_file_name = os.path.join(pth, "yolov7/yolov7-tiny.weights")


    class_names = []
    with open(class_file_name, "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]

    # vc = cv2.VideoCapture("demo.mp4")
    # vc = cv2.VideoCapture(0)

    net = cv2.dnn.readNet(weight_file_name, cfg_file_name)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)



    # net = cv2.dnn.readNetFromTensorflow(tp_weight_file_name, tp_cfg_file_name)
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_DEFAULT)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(288, 288), scale= 1/255, swapRB=True)
    #model.setInputParams(size=(416, 416), scale= 1/255, swapRB=True)


    vs = VideoStream().start()

    while cv2.waitKey(1) < 1:
        # (grabbed, frame) = vc.read()
        # if not grabbed:
        #     exit()

        frame = vs.read()
        frame = imutils.resize(frame, width=800)

        start = time.time()
        classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
        end = time.time()

        #print(classes, scores, boxes)

        start_drawing = time.time()
        for (classid, score, box) in zip(classes, scores, boxes):
            if classid == 0:
                color = COLORS[int(classid) % len(COLORS)]
                label = "%s : %f" % (class_names[classid], score)
                cv2.rectangle(frame, box, color, 2)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        end_drawing = time.time()

        fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % \
        (1 / (end - start), (end_drawing - start_drawing) * 1000)
        cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("Yolov7 detections", frame)


if __name__ == '__main__':
    run_stream()

    # t1 = threading.Thread(run_stream())
    # t1.daemon = True
    #
    # t1.start()

    # p = Process(target=run_stream(), args=())
    # p.start()


