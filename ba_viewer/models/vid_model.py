import cv2
import numpy as np
import pandas as pd
from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt
from PySide6.QtGui import QColor

from ba_viewer.utils.constants import BEHAV_ACTUAL_COL, BEHAV_COLUMN_NAMES, VALUE2COLOR


class VidModel:

    vid: cv2.VideoCapture
    fps = int
    nframes = int
    jump_size: int

    def __init__(self, *args, **kwargs):
        self.vid = None
        self.fps = None
        self.nframes = None
        self.jump_size = None

    def load(self, fp: str):
        # Read video
        self.vid = cv2.VideoCapture(fp)
        if not self.vid.isOpened():
            raise ValueError("Error opening video")
        # Get video properties
        self.fps = int(self.vid.get(cv2.CAP_PROP_FPS))
        self.nframes = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.jump_size = int(5 * self.fps)

    def read(self) -> tuple[bool, np.ndarray]:
        if not isinstance(self.vid, cv2.VideoCapture):
            return False, np.array([])
        return self.vid.read()

    # def set_frame(self, frame_num: int) -> None:
    #     self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
