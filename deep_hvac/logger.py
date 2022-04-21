import logging
import sys

root = logging.getLogger('deep_hvac')
root.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


logger = logging.getLogger('deep_hvac')


def debug(*args, **kwargs):
    logger.debug(*args, **kwargs)


def log(*args, **kwargs):
    logger.info(*args, **kwargs)
