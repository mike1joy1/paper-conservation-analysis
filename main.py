
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from paperdiff.loading import load_image
from paperdiff.stats import verdict_for_region
from paperdiff.workflow import describe_region_change, evaluate_reference_pair


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Paper Conservation Analysis")
        self.resize(1200, 800)

        self.b1_path = None
        self.b2_path = None
        self.a_path = None

        self.status_label = QLabel("Load B1, B2 and A to compare the already aligned images.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 15px; margin: 8px;")

        self.load_b1_button = QPushButton("Load B1")
        self.load_b2_button = QPushButton("Load B2")
        self.load_a_button = QPushButton("Load A")
        self.analyze_button = QPushButton("Analyse")

        self.load_b1_button.clicked.connect(self.load_b1)
        self.load_b2_button.clicked.connect(self.load_b2)
        self.load_a_button.clicked.connect(self.load_a)
        self.analyze_button.clicked.connect(self.analyze_images)

        self.b1_preview = QLabel("B1")
        self.b1_preview.setAlignment(Qt.AlignCenter)
        self.b1_preview.setMinimumSize(260, 210)
        self.b1_preview.setStyleSheet("border: 1px solid #888; background: #f3f3f3;")

        self.b2_preview = QLabel("B2")
        self.b2_preview.setAlignment(Qt.AlignCenter)
        self.b2_preview.setMinimumSize(260, 210)
        self.b2_preview.setStyleSheet("border: 1px solid #888; background: #f3f3f3;")

        self.a_preview = QLabel("A")
        self.a_preview.setAlignment(Qt.AlignCenter)
        self.a_preview.setMinimumSize(260, 210)
        self.a_preview.setStyleSheet("border: 1px solid #888; background: #f3f3f3;")

        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("Analysis summary will appear here.")

        button_row = QHBoxLayout()
        button_row.addWidget(self.load_b1_button)
        button_row.addWidget(self.load_b2_button)
        button_row.addWidget(self.load_a_button)
        button_row.addWidget(self.analyze_button)

        preview_row = QHBoxLayout()
        preview_row.addWidget(self.b1_preview)
        preview_row.addWidget(self.b2_preview)
        preview_row.addWidget(self.a_preview)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.status_label)
        main_layout.addLayout(button_row)
        main_layout.addLayout(preview_row)
        main_layout.addWidget(self.report)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def _set_preview_from_path(self, label: QLabel, path: str | Path):
        image = load_image(path)
        self._set_preview_from_array(label, image)

    def _set_preview_from_array(self, label: QLabel, image):
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        array = np.clip(array, 0.0, 1.0)
        rgb = (array * 255.0).astype("uint8")
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def load_b1(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select B1 image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if path:
            self.b1_path = path
            self._set_preview_from_path(self.b1_preview, path)
            self.status_label.setText(f"B1 loaded: {path}")

    def load_b2(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select B2 image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if path:
            self.b2_path = path
            self._set_preview_from_path(self.b2_preview, path)
            self.status_label.setText(f"B2 loaded: {path}")

    def load_a(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select A image", "", "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)")
        if path:
            self.a_path = path
            self._set_preview_from_path(self.a_preview, path)
            self.status_label.setText(f"A loaded: {path}")

    def analyze_images(self):
        if self.b1_path is None or self.b2_path is None or self.a_path is None:
            QMessageBox.warning(self, "Missing input", "Load B1, B2, and A before running the analysis.")
            return

        try:
            b1 = load_image(self.b1_path)
            b2 = load_image(self.b2_path)
            a = load_image(self.a_path)
            results = evaluate_reference_pair(b1, b2, a)
            base = results["base"]
            test = results["test"]

            summary_lines = [
                "Base comparison: B1 vs B2",
                f"Mean overall difference: {base['score']:.3f} ΔE00",
                f"Paper verdict: {verdict_for_region(base['paper'])}",
                f"Ink verdict: {verdict_for_region(base['ink'])}",
                "",
                describe_region_change(base["paper"], "paper"),
                describe_region_change(base["ink"], "ink"),
                "",
                "Test comparison: B1 vs A",
                f"Mean overall difference: {test['score']:.3f} ΔE00",
                f"Paper verdict: {verdict_for_region(test['paper'])}",
                f"Ink verdict: {verdict_for_region(test['ink'])}",
                "",
                describe_region_change(test["paper"], "paper"),
                describe_region_change(test["ink"], "ink"),
                "",
                "Raw metric values:",
            ]
            for key, value in base["paper"].items():
                summary_lines.append(f"- B2 paper {key}: {value:.3f}")
            for key, value in base["ink"].items():
                summary_lines.append(f"- B2 ink {key}: {value:.3f}")
            for key, value in test["paper"].items():
                summary_lines.append(f"- A paper {key}: {value:.3f}")
            for key, value in test["ink"].items():
                summary_lines.append(f"- A ink {key}: {value:.3f}")

            self.report.setPlainText("\n".join(summary_lines))
            self.status_label.setText("Analysis complete. The report below compares B1 against both B2 and A using the current paper and ink masks.")
        except Exception as exc:
            QMessageBox.critical(self, "Analysis failed", f"Unable to analyze the images:\n{exc}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
