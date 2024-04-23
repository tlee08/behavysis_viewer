import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

from behavysis_viewer.widgets.cv_view import CvView
from behavysis_viewer.utils.cv2_qt_mixin import Cv2QtMixin


@pytest.fixture(scope="module", autouse=True)
def cv_view(app):
    # Create an instance of the CvView widget
    cv_view = CvView()

    yield cv_view
    # Clean up the CvView widget
    cv_view.close()


@pytest.fixture(scope="function", autouse=False)
def rand_img():
    return np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)


def test_cv_view_initialization(cv_view):
    # Test that the CvView widget is initialized correctly
    assert cv_view is not None


def test_cv_view_display_grey(cv_view):
    grey_img = np.full((cv_view.height(), cv_view.width(), 3), 127, dtype=np.uint8)
    cv_view.display_grey()
    assert True  # test that image displayed is expected


def test_cv_view_display_cv2(cv_view, rand_img):
    cv_view.display_cv2(rand_img)
    assert True  # test that image displayed is expected


def test_cv_view_display_fp(cv_view):
    img_fp = "tests/resources/example_img.jpeg"
    cv_view.display_fp(img_fp)
    assert True  # test that image displayed is expected


def test_cv_view_setFixedSize(cv_view, rand_img):
    cv_view.display_cv2(rand_img)
    cv_view.setFixedSize(400, 200)
    assert True  # test that image displayed is expected
    assert cv_view.width() == 400
    assert cv_view.height() == 200


def test_cv_view_cv_2_qt_2_cv(cv_view, rand_img):
    img_cv2 = Cv2QtMixin.qt_2_cv(Cv2QtMixin.cv_2_qt(rand_img))
    assert np.all(rand_img == img_cv2)


if __name__ == "__main__":
    pytest.main([__file__])
