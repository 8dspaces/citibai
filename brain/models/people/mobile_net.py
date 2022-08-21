import imutils
import cv2
import os, time
import tensorflow as tf
from brain.models.people.object_detection.utils import label_map_util
from brain.models.people.object_detection.utils import visualization_utils as vis_utils
import numpy as np


class MobileNetV1:

    NUM_CLASSES = 90
    IMAGE_SIZE = (12, 8)

    CONFIDENCE_THRESHOLD = 0.5  # 0.35

    def __init__(self):

        pth = os.path.dirname(__file__)

        MODEL_NAME = os.path.join(pth, 'mobilenet_v1')  # fast
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
        PATH_TO_LABELS = os.path.join(MODEL_NAME, 'data', 'mscoco_label_map.pbtxt')

        # LOAD GRAPH
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(
                                    self.label_map,
                                    max_num_classes=self.NUM_CLASSES,
                                    use_display_name=True)

        self.category_index = label_map_util.create_category_index(self.categories)

        # print('Finish Load Graph..')
        # print(type(category_index))
        # print("dict['Name']: ", category_index[1]['name'])

    def update(self):
        pass



