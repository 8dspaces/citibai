import pyttsx3
import threading
import time

from brain import events
from brain.log import logger
from brain.words import sounds


class DummySpeaker(threading.Thread):

    def __init__(self):

        threading.Thread.__init__(self)
        self.engine = pyttsx3.init()

        #self.engine = pyttsx3.init("espeak", True)  #for Pi
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', 'zh')


    def run(self):

        self.engine = pyttsx3.init()

        while True:

            self.engine.say(self.get_word(events.speaker_queue.get()))
            self.engine.runAndWait()

            ## needed for macos, no need for pi
            time.sleep(2)
            self.engine.endLoop()
            self.engine.stop()

            events.speaker_queue.task_done()

    def get_word(self, name):

        return sounds.get(name, "我无语了")


if __name__ == '__main__':

    s = DummySpeaker()

    s.start()
    events.speaker_queue.put("ready")
    for i in range(0, 2):
        events.speaker_queue.put("thanks")

    events.speaker_queue.join()

    s.join()


