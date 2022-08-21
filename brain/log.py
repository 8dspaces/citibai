import pathlib
import os
import sys
from loguru import logger

LOG_DIR = pathlib.Path(__file__).parent.parent.absolute()

LOG_FILE = LOG_DIR.joinpath('log/citibai_log_{time:YYYY-MM-DD}.log')
logger.add(LOG_FILE, rotation="5MB", format="{time:HH:mm:ss.SS} | {level} | {thread} | {message}")

