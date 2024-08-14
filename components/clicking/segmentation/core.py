# components/mousetrap/fetch_data/core.py
import random
import time

from clicking.segmentation.sam2 import SAM2Model


def fetch_data() -> list:
    """We will fetch a random number of geos"""
    data = [1, 2, 3, 4, 5]
    return data