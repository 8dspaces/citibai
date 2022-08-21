import numpy as np
import cv2
import os, time
import tensorflow as tf
from brain.models.people.object_detection.utils import label_map_util
from brain.models.people.object_detection.utils import visualization_utils as vis_utils
#import ipywidgets.widgets as widgets
#from brain.models.people.utils.image_fun import bgr8_to_jpeg

import imutils
from imutils.video import VideoStream

import time


def func():
    # print("start")
    # cap = cv2.VideoCapture(0)
    # cap.set(3, 320)  # set Width
    # cap.set(4, 240)  # set Height
    # cap.set(5, 30)  # 设置帧率
    #
    # print("cap")

    #image_widget = widgets.Image(format='jpg', width=320, height=240)
    # Init tf model

    pth = os.path.dirname(__file__)

    # class_file_name = os.path.join(pth, "yolov7/coco.names")
    # weight_file_name = os.path.join(pth, "yolov7/yolov7-tiny.cfg")
    # cfg_file_name = os.path.join(pth, "yolov7/yolov7-tiny.weights")

    # MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09' #fast

    MODEL_NAME = os.path.join(pth, 'mobilenet_v1')  # fast
    PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
    PATH_TO_LABELS = os.path.join(MODEL_NAME, 'data', 'mscoco_label_map.pbtxt')

    NUM_CLASSES = 90
    IMAGE_SIZE = (12, 8)
    fileAlreadyExists = os.path.isfile(PATH_TO_CKPT)

    if not fileAlreadyExists:
        print('Model does not exsist !')
        exit

    # LOAD GRAPH
    print('Loading...')
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    print('Finish Load Graph..')

    print(type(category_index))
    print("dict['Name']: ", category_index[1]['name'])

    vs = VideoStream().start()


    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            while cv2.waitKey(1) < 1:
                frame = vs.read()
                frame = imutils.resize(frame, width=800)

                # Main
                t_start = time.time()
                #fps = 0

                # frame = cv2.flip(frame, -1) # Flip camera vertically
                # frame = cv2.resize(frame,(320,240))
                ##############
                image_np_expanded = np.expand_dims(frame, axis=0)
                image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
                detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = detection_graph.get_tensor_by_name('num_detections:0')

                #             print('Running detection..')
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                #             print('Done.  Visualizing..')
                vis_utils.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)
                count_person = 0
                for i in range(0, 10):
                    if scores[0][i] >= 0.5:
                        print(category_index[int(classes[0][i])]['name'])
                        if category_index[int(classes[0][i])]['name'] == "person":
                            count_person += 1
                print("Person count : {}".format(count_person))
                if count_person >= 2:
                    print("whilse")
                ##############
                # fps = fps + 3
                # mfps = fps / (time.time() - t_start)

                fps = 1 / (time.time() - t_start)

                cv2.putText(frame, "FPS " + str(int(fps)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                cv2.imshow("MobileNet detections", frame)


if __name__ == '__main__':

    func()