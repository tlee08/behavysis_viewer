"""
_summary_
"""

import sys

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

from behavysis_viewer.utils.cv2_qt_mixin import Cv2QtMixin


class CvView(QLabel):
    """_summary_"""

    def __init__(self, parent=None, **kwargs):
        """
        Displays cv2 images (i.e. numpy arrays).

        Parameters
        ----------
        parent : _type_, optional
            _description_, by default None
        """
        super().__init__(parent)

        self.display_grey()

    def display_grey(self) -> None:
        """Makes a grey image (like background)."""
        img_cv = np.full((self.height(), self.width(), 3), 127, dtype=np.uint8)
        self.display_cv2(img_cv)

    def display_fp(self, fp: str):
        """
        Displaying image in given `fp`.

        Parameters
        ----------
        fp : str
            _description_
        """
        img_cv = cv2.imread(fp)
        if img_cv is not None:
            self.display_cv2(img_cv)

    def display_cv2(self, img_cv: np.ndarray) -> None:
        """
        Display an opencv image on the widget.
        in the instance for future reference.

        Parameters
        ----------
        img_cv : np.ndarray
            _description_
        """
        # Resize to widget size
        w, h = self.width(), self.height()
        img_cv = cv2.resize(img_cv, (w, h), interpolation=cv2.INTER_AREA)
        # Colour formatting img_cv
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
        # img_cv to QPixmap
        img_qt = Cv2QtMixin.cv_2_qt(img_cv)
        # QImage to QPixmap and displkaying on widget
        self.setPixmap(QPixmap.fromImage(img_qt))

    def setFixedSize(self, width, height):
        super().setFixedSize(width, height)
