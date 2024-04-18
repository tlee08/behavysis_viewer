import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session", autouse=True)
def app():
    # Set up the QApplication
    app = QApplication([])
    # Show the widget
    yield app
    # Tear down the QApplication
    app.quit()
    app.deleteLater()
