import os
import time
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('root')
logger.setLevel(logging.DEBUG)


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Logger(metaclass=Singleton):
    def __init__(self, filepath, suffix):
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        self.filepath = os.path.join(filepath, time.strftime("%Y%m%d%H%M%S") + "-" + suffix + ".txt")

    def log(self, text):
        with open(self.filepath, "a+") as f:
            f.write(text + "\r\n")

    def info(self, text):
        logging.info(text)

    def debug(self, text):
        logging.debug(text)

