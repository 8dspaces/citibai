import contextlib
import time
import threading
import uvicorn
from uvicorn import Config

from web.app import stream

from brain import events
from brain.log import logger
from speaker.google import GoogleSpeaker as Speaker
from car.core import SmartCar

class CitiDabai(uvicorn.Server):

    def install_signal_handlers(self):
        pass

    @contextlib.contextmanager
    def run_in_thread(self):
        thread = threading.Thread(target=self.run)
        thread.start()
        try:
            while not self.started:
                time.sleep(1e-3)
            yield
        finally:
            self.should_exit = True
            thread.join()

config = Config("server:api_main", host="127.0.0.1", port=9017, log_level="info")
dabai = CitiDabai(config=config)

speaker = Speaker()
car = SmartCar()

if __name__ == '__main__':

    logger.info("this is from Main")

    speaker.daemon = True
    car.daemon = True

    speaker.start()
    car.start()

    with dabai.run_in_thread():

        logger.info("stream Server start")

        stream.start()
        events.speaker_queue.put("ready")

        events.speaker_queue.join()

        car.join()
        speaker.join()

