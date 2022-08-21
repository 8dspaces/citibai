import cv2
import mediapipe as mp
import time
import imutils
import cv2


class SimpleHandDetector:

    def __init__(self):

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.mpDraw = mp.solutions.drawing_utils
        self.handLmsStyle = self.mpDraw.DrawingSpec(color=(0, 0, 255), thickness=3)
        self.handConStyle = self.mpDraw.DrawingSpec(color=(0, 255, 0), thickness=5)
        # self.pTime = 0
        # self.cTime = 0

    def update(self, image):

        print("update")

    def detect(self, image):

        print("detect people")

        # 灰度处理
        gray = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)

        # 绘制矩形和圆形检测人脸
        num = 0

        # 检查人脸 按照1.1倍放到 周围最小像素为5
        face_zone = self.face_detect.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        cnts = imutils.grab_contours(face_zone)
        for x, y, w, h in cnts:
            num = num + 1
            cv2.rectangle(image, pt1=(x, y), pt2=(x + w, y + h), color=[0, 0, 255], thickness=2)
            cv2.circle(image, center=(x + w // 2, y + h // 2), radius=w // 2, color=[0, 255, 0], thickness=2)
            cv2.putText(image, str(num), (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

        return face_zone



