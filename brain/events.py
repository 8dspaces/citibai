from threading import Event
from queue import Queue


target_mark = Event(),
car_stop = Event(),
speaker_queue = Queue(1)

pose_flag = False