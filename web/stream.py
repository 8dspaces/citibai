import copy
import threading
from collections import Counter

import imutils
import time
import cv2
import tensorflow as tf
import numpy as np

# switch on different Model of People detect
from brain.models.people.mobile_net import MobileNetV1
from brain.models.people.yolo7_people import PeopleDetectorV7

# switch on different Model of Hand
from brain.models.hand.pose_detect import PoseDetector
from brain.models.hand.simple_detect import SimpleHandDetector
from imutils.video import VideoStream

from brain.models.people.object_detection.utils import visualization_utils as vis_utils


class StreamServer:

    detect_model_mn = True

    def __init__(self):

        self.outputFrame_people = None
        self.outputFrame_hand = None
        self.outputFrame_raw = None

        self.lock_people = threading.Lock()
        self.lock_hand = threading.Lock()
        self.flip = True

        if self.detect_model_mn:
            self.t1 = threading.Thread(target=self.detect_people_mn, args=())
        else:
            self.t1 = threading.Thread(target=self.detect_people_yolo, args=())

        # initialize the video stream and allow the camera sensor to warmup
        self.vs = VideoStream().start()
        # self.vs = VideoStream("http://192.168.50.1:8080/?action=stream").start()
        time.sleep(2.0)

    def start(self):

        # start a thread that will perform motion detection

        self.t1.daemon = True
        self.t1.start()

        #self.t2.daemon = True
        #self.t2.start()

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

    def detect_people_yolo(self, width=800):

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

            person_count = 0
            for (classid, score, box) in zip(classes, scores, boxes):
                if classid == 0:
                    color = pd.COLORS[int(classid) % len(pd.COLORS)]
                    label = "%s : %f" % (pd.class_names[classid], score)
                    cv2.rectangle(frame, box, color, 2)
                    cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                if classid == 0 and score > pd.CONFIDENCE_THRESHOLD:
                    person_count += 1

            fps_label = "FPS: %.2f / person: %d" % (1 / (end - start), person_count)
            cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            #cv2.imshow("detections", frame)

            with self.lock_people:
                self.outputFrame_people = frame.copy()

    def detect_simple_hand(self, width=800):

        shd = SimpleHandDetector()
        pTime = 0
        cTime = 0

        while True:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=width)
            if self.flip:
                frame = cv2.flip(frame, 1)

            #self.outputFrame_raw = frame.copy()

            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = shd.hands.process(imgRGB)

            # print(result.multi_hand_landmarks)
            imgHeight = frame.shape[0]
            imgWidth = frame.shape[1]

            if result.multi_hand_landmarks:
                for handLms in result.multi_hand_landmarks:
                    shd.mpDraw.draw_landmarks(frame, handLms, shd.mpHands.HAND_CONNECTIONS,
                                               shd.handLmsStyle, shd.handConStyle)
                    print("get picture")

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            cv2.putText(frame, f"FPS : {int(fps)}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)


            with self.lock_hand:
                self.outputFrame_hand = frame.copy()

    def detect_hand_pose(self, width=800):

        pd = PoseDetector()

        while True:
            frame = self.vs.read()
            frame = imutils.resize(frame, width=width)
            if self.flip:
                frame = cv2.flip(frame, 1)

            fps = pd.cvFpsCalc.get()

            key = cv2.waitKey(10)
            if key == 27:  # ESC
                break

            number, mode = pd.select_mode(key, pd.mode)

            debug_image = copy.deepcopy(frame)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame.flags.writeable = False
            results = pd.hands.process(frame)
            frame.flags.writeable = True

            #  ####################################################################
            if results.multi_hand_landmarks is not None:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                      results.multi_handedness):
                    brect = pd.calc_bounding_rect(debug_image, hand_landmarks)

                    landmark_list = pd.calc_landmark_list(debug_image, hand_landmarks)

                    pre_processed_landmark_list = pd.pre_process_landmark(
                        landmark_list)
                    pre_processed_point_history_list = pd.pre_process_point_history(
                        debug_image, pd.point_history)

                    pd.logging_csv(number, mode, pre_processed_landmark_list,
                                pre_processed_point_history_list)

                    hand_sign_id = pd.keypoint_classifier(pre_processed_landmark_list)
                    if hand_sign_id == 2:
                        pd.point_history.append(landmark_list[8])
                    else:
                        pd.point_history.append([0, 0])

                    finger_gesture_id = 0
                    point_history_len = len(pre_processed_point_history_list)
                    if point_history_len == (pd.history_length * 2):
                        finger_gesture_id = pd.point_history_classifier(
                            pre_processed_point_history_list)

                    pd.finger_gesture_history.append(finger_gesture_id)
                    most_common_fg_id = Counter(
                        pd.finger_gesture_history).most_common()

                    debug_image = pd.draw_bounding_rect(pd.use_brect, debug_image, brect)

                    #debug_image = pd.draw_landmarks(debug_image, landmark_list)  ## fast
                    debug_image = pd.draw_landmarks_slow(debug_image, hand_landmarks) # slow

                    debug_image = pd.draw_info_text(
                        debug_image,
                        brect,
                        handedness,
                        pd.keypoint_classifier_labels[hand_sign_id],
                        pd.point_history_classifier_labels[most_common_fg_id[0][0]],
                    )
            else:
                pd.point_history.append([0, 0])

            debug_image = pd.draw_point_history(debug_image, pd.point_history)
            debug_image = pd.draw_info(debug_image, fps, mode, number)

            with self.lock_hand:
                self.outputFrame_hand = debug_image.copy()

            # cv.imshow('Hand Gesture Recognition', debug_image)

    # Output Stream
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

    def generate_hand(self):
        # grab global references to the output frame and lock variables global outputFrame, lock
        # loop over frames from the output stream
        while True:
            # wait until the lock is acquired

            with self.lock_hand:
                # check if the output frame is available, otherwise skip
                # the iteration of the loop
                if self.outputFrame_hand is None:
                    continue
                # encode the frame in JPEG format
                (flag, encodedImage) = cv2.imencode(".jpg", self.outputFrame_hand)
                # ensure the frame was successfully encoded
                if not flag:
                    continue

            # yield the output frame in the byte format
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
                   bytearray(encodedImage) + b'\r\n')

    def generate_raw(self):
        # grab global references to the output frame and lock variables global outputFrame, lock
        # loop over frames from the output stream
        while True:
            # wait until the lock is acquired
            with self.lock:
                # check if the output frame is available, otherwise skip
                # the iteration of the loop
                if self.outputFrame_people is None:
                    continue
                # encode the frame in JPEG format
                (flag, encodedImage) = cv2.imencode(".jpg", self.outputFrame_raw)
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
        self.t2.join()


if __name__ == '__main__':

    pass