import logging
import sys

root = logging.getLogger()
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


logger = logging.getLogger()


def debug(*args, **kwargs):
    logger.debug(*args, **kwargs)


def log(*args, **kwargs):
    logger.log(*args, **kwargs)