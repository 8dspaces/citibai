import pyttsx3
import threading
import pathlib
import os
import time

from brain import events
from brain.log import logger
from brain.words import sounds
#from speaker.utils import play_audio
# import playsound

#from speaker.player import play_audio_background
from speaker.playerLite import play_audio_background


class GoogleSpeaker(threading.Thread):

    def __init__(self):

        threading.Thread.__init__(self)

        self.lock = threading.Lock()

    def run(self):

        logger.info("this is from google speaker")

        while True:

            word = events.speaker_queue.get()
            file = self.get_audio(word)
            with self.lock:
                play_audio_background(file)

            events.speaker_queue.task_done()

    def get_audio(self, name, pth="gtts"):

        audio_dir = pathlib.Path(__file__).parent.absolute().joinpath('audio/{}'.format(pth))
        return audio_dir.joinpath(name + '.mp3')

if __name__ == '__main__':

    s = GoogleSpeaker()
    s.start()

    events.speaker_queue.put("ready")
    for i in range(0, 2):
        events.speaker_queue.put("thanks")

    events.speaker_queue.join()
    s.join()


