import threading
import time
import cv2
import tensorflow as tf
import numpy as np

import imutils
from imutils.video import VideoStream

from brain.models.people.mobile_net import MobileNetV1
from brain.models.people.yolo7_people import PeopleDetectorV7

from brain.models.people.object_detection.utils import visualization_utils as vis_utils


## for Test purpose
class StreamServer:

    def __init__(self):

        # initialize the output frame and a lock used to ensure thread-safe exchanges of the
        # output frames (useful when multiple browsers/tabs are viewing the stream)

        self.outputFrame_people = None
        self.lock_people = threading.Lock()
        self.flip = True

        #self.t1 = threading.Thread(target=self.detect_people_mn, args=())
        self.t1 = threading.Thread(target=self.detect_people_mn, args=())

        self.vs = VideoStream().start()
        # self.vs = VideoStream("http://192.168.50.1:8080/?action=stream").start()
        time.sleep(2.0)

    def start(self):

        #self.t1.daemon = True
        self.t1.start()

    def detect_people(self, width=800):

        pd = PeopleDetectorV7()

        while True:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=width)
            if self.flip:
                frame = cv2.flip(frame, 1)

            #self.outputFrame_raw = frame.copy()

            start = time.time()

            classes, scores, boxes = pd.model.detect(frame, pd.CONFIDENCE_THRESHOLD, pd.NMS_THRESHOLD)
            end = time.time()

            #print(classes, scores, boxes)

            start_drawing = time.time()

            person_count = 0
            for (classid, score, box) in zip(classes, scores, boxes):
                if classid == 0:
                    color = pd.COLORS[int(classid) % len(pd.COLORS)]
                    label = "%s : %f" % (pd.class_names[classid], score)
                    cv2.rectangle(frame, box, color, 2)
                    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if classid == 0 and score > pd.CONFIDENCE_THRESHOLD:
                    person_count += 1

            end_drawing = time.time()

            fps_label = "FPS: %.2f / person: %d" % (1 / (end - start), person_count)
            cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            #cv2.imshow("detections", frame)

            with self.lock_people:
                self.outputFrame_people = frame.copy()

    def detect_people_mn(self, width=800):

        mn = MobileNetV1()

        with mn.detection_graph.as_default():
            with tf.compat.v1.Session(graph=mn.detection_graph) as sess:
                while True:

                    frame = self.vs.read()
                    frame = imutils.resize(frame, width=width)

                    # Main
                    t_start = time.time()

                    image_np_expanded = np.expand_dims(frame, axis=0)
                    image_tensor = mn.detection_graph.get_tensor_by_name('image_tensor:0')
                    detection_boxes = mn.detection_graph.get_tensor_by_name('detection_boxes:0')
                    detection_scores = mn.detection_graph.get_tensor_by_name('detection_scores:0')
                    detection_classes = mn.detection_graph.get_tensor_by_name('detection_classes:0')
                    num_detections = mn.detection_graph.get_tensor_by_name('num_detections:0')

                    # print('Running detection..')
                    (boxes, scores, classes, num) = sess.run(
                        [detection_boxes, detection_scores, detection_classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})

                    # print('Done.  Visualizing..')
                    vis_utils.visualize_boxes_and_labels_on_image_array(
                        frame,
                        np.squeeze(boxes),
                        np.squeeze(classes).astype(np.int32),
                        np.squeeze(scores),
                        mn.category_index,
                        use_normalized_coordinates=True,
                        line_thickness=2)

                    count_person = 0
                    for i in range(0, 10):
                        if scores[0][i] >= mn.CONFIDENCE_THRESHOLD:
                            #print(mn.category_index[int(classes[0][i])]['name'])
                            if mn.category_index[int(classes[0][i])]['name'] == "person":
                                count_person += 1

                    fps = 1 / (time.time() - t_start)
                    #print("Person count : {}".format(count_person))
                    if count_person >= 2:
                        print("whistle")

                    labels = "PFS: {}| People: {}".format(fps, count_person)
                    cv2.putText(frame, labels, (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                    # cv2.imshow("detections", frame)
                    with self.lock_people:
                        self.outputFrame_people = frame.copy()

    def generate_people(self):
        # grab global references to the output frame and lock variables global outputFrame, lock
        # loop over frames from the output stream
        while True:
            # wait until the lock is acquired
            with self.lock_people:
                # check if the output frame is available, otherwise skip
                # the iteration of the loop
                if self.outputFrame_people is None:
                    continue
                # encode the frame in JPEG format
                (flag, encodedImage) = cv2.imencode(".jpg", self.outputFrame_people)
                # ensure the frame was successfully encoded
                if not flag:
                    continue

            # yield the output frame in the byte format
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) + b'\r\n')

    def close(self):
        # release the video stream pointer
        self.vs.stop()
        self.t1.jon()


if __name__ == '__main__':

    s = StreamServer()
    s.start()


