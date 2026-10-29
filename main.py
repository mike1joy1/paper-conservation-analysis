
import sys

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Paper Conservation Analysis")
        self.resize(800, 500)

        message = QLabel("Hello! Paper Conservation Analysis is running.")
        message.setStyleSheet("font-size: 18px; padding: 20px;")

        self.setCentralWidget(message)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())
