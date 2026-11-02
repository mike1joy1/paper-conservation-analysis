"""Desktop app for paper conservation analysis.

The workflow is intentionally simple: the images are assumed to have been aligned
and cropped before they reach this application, and the app focuses on the
scientific comparison and report generation.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

# Add the source tree to the import path so the app can import the package.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from paperdiff.loading import load_image
from paperdiff.stats import verdict_for_region
from paperdiff.workflow import describe_region_change, evaluate_reference_pair


class ClickableLabel(QLabel):
    """A QLabel that emits a click signal so the alignment viewer can cycle images."""

    clicked = Signal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    """The main window containing the five-step analysis workflow."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Paper Conservation Analysis")
        self.resize(1200, 800)

        # The scientific workflow assumes the images are already aligned and cropped
        # before they are opened in this app.
        self.image_paths = {"B1": None, "B2": None, "A": None}
        self.loaded_arrays = {"B1": None, "B2": None, "A": None}
        self.mask_arrays = {"B1": None, "B2": None, "A": None}
        self.current_alignment_view = "B1"

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.load_tab = self._create_load_tab()
        self.check_alignment_tab = self._create_check_alignment_tab()
        self.mask_tab = self._create_mask_tab()
        self.analysis_options_tab = self._create_analysis_options_tab()
        self.results_tab = self._create_results_tab()

        self.tab_widget.addTab(self.load_tab, "1) Load images")
        self.tab_widget.addTab(self.check_alignment_tab, "2) Check alignment")
        self.tab_widget.addTab(self.mask_tab, "3) Mask")
        self.tab_widget.addTab(self.analysis_options_tab, "4) Analysis options")
        self.tab_widget.addTab(self.results_tab, "5) Results")

    def _preview_from_array(self, label: QLabel, image):
        """Convert an image array into a Qt pixmap for display in the app."""
        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)

        if array.size == 0:
            label.setText("No image")
            label.setPixmap(QPixmap())
            return

        # Most image loaders in this project provide values in the 0..1 range.
        array = np.clip(array, 0.0, 1.0)
        rgb = (array * 255.0).astype("uint8")
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _display_mask_preview(self, label: QLabel, mask_array, border_mask=None):
        """Render a black/white mask with red border pixels for the mask edge."""
        if mask_array is None:
            label.setText("No mask")
            label.setPixmap(QPixmap())
            return

        mask = np.asarray(mask_array, dtype=bool)
        if border_mask is None:
            border_mask = mask ^ binary_erosion(mask, iterations=1)

        h, w = mask.shape
        rendered = np.zeros((h, w, 3), dtype=np.uint8)
        rendered[mask] = [255, 255, 255]
        rendered[border_mask] = [255, 0, 0]

        qimage = QImage(rendered.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())
        label.setPixmap(pixmap.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _set_preview_text(self, label: QLabel, text: str):
        label.setText(text)
        label.setPixmap(QPixmap())

    def _open_image_for_key(self, key: str):
        """Open a single image and store it in the loaded arrays for the given key."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {key} image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return

        self.image_paths[key] = path
        image = load_image(path)
        self.loaded_arrays[key] = image

        preview_label = self.load_preview_labels[key]
        self._preview_from_array(preview_label, image)
        self.path_labels[key].setText(path)

        if self.current_alignment_view == key:
            self._display_alignment_image()

    def _create_load_tab(self):
        """Tab 1: load the three images and show their file paths."""
        page = QWidget()
        layout = QVBoxLayout(page)

        input_group = QGroupBox("Image type")
        input_layout = QHBoxLayout(input_group)
        self.image_type_group = QButtonGroup(page)
        self.colour_radio = QRadioButton("Colour images")
        self.normal_map_radio = QRadioButton("Normal maps")
        self.colour_radio.setChecked(True)
        self.image_type_group.addButton(self.colour_radio)
        self.image_type_group.addButton(self.normal_map_radio)
        input_layout.addWidget(self.colour_radio)
        input_layout.addWidget(self.normal_map_radio)
        layout.addWidget(input_group)

        self.load_preview_labels = {}
        self.path_labels = {}
        grid = QGridLayout()
        for col, key in enumerate(["B1", "B2", "A"]):
            card = QWidget()
            card_layout = QVBoxLayout(card)

            preview_label = QLabel(f"{key}")
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setMinimumSize(240, 180)
            preview_label.setStyleSheet("border: 1px solid #888; background: #262626;")
            card_layout.addWidget(preview_label)

            button = QPushButton(f"Load {key}")
            button.clicked.connect(lambda checked, value=key: self._open_image_for_key(value))
            card_layout.addWidget(button)

            path_label = QLabel("No file selected")
            path_label.setWordWrap(True)
            card_layout.addWidget(path_label)

            self.load_preview_labels[key] = preview_label
            self.path_labels[key] = path_label
            grid.addWidget(card, 0, col)

        layout.addLayout(grid)
        return page

    def _display_alignment_image(self):
        """Show one of the three images in the alignment check area."""
        key = self.current_alignment_view
        image = self.loaded_arrays.get(key)
        if image is None:
            self.alignment_preview.setText(f"{key} not loaded")
            self.alignment_preview.setPixmap(QPixmap())
            return
        self._preview_from_array(self.alignment_preview, image)

    def _cycle_alignment_image(self):
        order = ["B1", "B2", "A"]
        current_index = order.index(self.current_alignment_view)
        self.current_alignment_view = order[(current_index + 1) % len(order)]
        self._update_alignment_buttons()
        self._display_alignment_image()

    def _update_alignment_buttons(self):
        for key, button in self.alignment_buttons.items():
            button.setChecked(key == self.current_alignment_view)

    def _create_check_alignment_tab(self):
        """Tab 2: visually inspect the alignment by switching the view between B1/B2/A."""
        page = QWidget()
        layout = QVBoxLayout(page)

        self.alignment_checkbox = QCheckBox("Images are aligned")
        self.alignment_checkbox.setChecked(False)
        layout.addWidget(self.alignment_checkbox)

        button_row = QHBoxLayout()
        self.alignment_buttons = {}
        for key in ["B1", "B2", "A"]:
            button = QPushButton(key)
            button.setCheckable(True)
            button.clicked.connect(lambda checked, value=key: self._set_alignment_view(value))
            button_row.addWidget(button)
            self.alignment_buttons[key] = button
        layout.addLayout(button_row)

        self.alignment_preview = ClickableLabel("B1")
        self.alignment_preview.setAlignment(Qt.AlignCenter)
        self.alignment_preview.setMinimumSize(600, 450)
        self.alignment_preview.setStyleSheet("border: 1px solid #888; background: #262626;")
        self.alignment_preview.clicked.connect(self._cycle_alignment_image)
        layout.addWidget(self.alignment_preview)

        self._update_alignment_buttons()
        return page

    def _set_alignment_view(self, key: str):
        self.current_alignment_view = key
        self._update_alignment_buttons()
        self._display_alignment_image()

    def _generate_mask_for_key(self, key: str):
        """Create a brightness mask for one image and display it with a red border."""
        image = self.loaded_arrays.get(key)
        if image is None:
            QMessageBox.warning(self, "Missing image", f"Load {key} before generating a mask.")
            return

        array = np.asarray(image)
        if array.ndim == 3:
            brightness = np.mean(array, axis=-1)
        else:
            brightness = array

        brightness = brightness.astype(np.float32)
        if brightness.max() <= 1.0:
            brightness = brightness * 255.0

        threshold = self.mask_threshold_slider.value()
        mask = brightness <= threshold

        # Keep the border simple and explicit: the mask front is the region, while the
        # border is the outer rim of that region. This makes it easy to inspect in the UI.
        border_pixels = self.mask_border_slider.value()
        if border_pixels > 0:
            border_mask = mask ^ binary_erosion(mask, iterations=border_pixels)
        else:
            border_mask = np.zeros_like(mask, dtype=bool)

        self.mask_arrays[key] = mask
        self.mask_border_arrays[key] = border_mask

        if self.same_mask_checkbox.isChecked():
            for other_key in ["B1", "B2", "A"]:
                self.mask_arrays[other_key] = mask
                self.mask_border_arrays[other_key] = border_mask

        for other_key in ["B1", "B2", "A"]:
            if self.mask_arrays.get(other_key) is not None:
                border_mask = self.mask_border_arrays.get(other_key)
                self._display_mask_preview(
                    self.mask_preview_labels[other_key],
                    self.mask_arrays[other_key],
                    border_mask,
                )

    def _create_mask_tab(self):
        """Tab 3: prepare and preview masks used for the paper and ink analysis."""
        page = QWidget()
        layout = QVBoxLayout(page)

        settings_group = QGroupBox("Mask settings")
        settings_layout = QVBoxLayout(settings_group)

        self.use_mask_checkbox = QCheckBox("Use mask")
        self.same_mask_checkbox = QCheckBox("Same mask")
        settings_layout.addWidget(self.use_mask_checkbox)
        settings_layout.addWidget(self.same_mask_checkbox)

        self.mask_threshold_slider = QSlider(Qt.Horizontal)
        self.mask_threshold_slider.setRange(0, 255)
        self.mask_threshold_slider.setValue(140)
        self.mask_threshold_label = QLabel("Mask threshold: 140")
        self.mask_threshold_slider.valueChanged.connect(
            lambda value: self.mask_threshold_label.setText(f"Mask threshold: {value}")
        )
        settings_layout.addWidget(self.mask_threshold_label)
        settings_layout.addWidget(self.mask_threshold_slider)

        self.mask_border_slider = QSlider(Qt.Horizontal)
        self.mask_border_slider.setRange(0, 20)
        self.mask_border_slider.setValue(2)
        self.mask_border_label = QLabel("Mask border: 2 px")
        self.mask_border_slider.valueChanged.connect(
            lambda value: self.mask_border_label.setText(f"Mask border: {value} px")
        )
        settings_layout.addWidget(self.mask_border_label)
        settings_layout.addWidget(self.mask_border_slider)

        layout.addWidget(settings_group)

        self.mask_preview_labels = {}
        self.mask_border_arrays = {"B1": None, "B2": None, "A": None}
        grid = QGridLayout()
        for col, key in enumerate(["B1", "B2", "A"]):
            card = QWidget()
            card_layout = QVBoxLayout(card)

            preview = QLabel(f"{key}")
            preview.setAlignment(Qt.AlignCenter)
            preview.setMinimumSize(240, 180)
            preview.setStyleSheet("border: 1px solid #888; background: #262626;")
            card_layout.addWidget(preview)

            button = QPushButton(f"Generate mask for {key}")
            button.clicked.connect(lambda checked, value=key: self._generate_mask_for_key(value))
            card_layout.addWidget(button)

            self.mask_preview_labels[key] = preview
            grid.addWidget(card, 0, col)

        layout.addLayout(grid)
        return page

    def _create_analysis_options_tab(self):
        """Tab 4: future analysis settings kept deliberately blank and simple."""
        page = QWidget()
        layout = QVBoxLayout(page)

        placeholder_group = QGroupBox("Future controls")
        placeholder_layout = QVBoxLayout(placeholder_group)

        self.future_checkbox_1 = QCheckBox("Use smoothing")
        self.future_checkbox_2 = QCheckBox("Include edge check")
        placeholder_layout.addWidget(self.future_checkbox_1)
        placeholder_layout.addWidget(self.future_checkbox_2)

        slider_1 = QSlider(Qt.Horizontal)
        slider_1.setRange(0, 100)
        slider_1.setValue(25)
        placeholder_layout.addWidget(QLabel("Sensitivity: 25"))
        placeholder_layout.addWidget(slider_1)

        slider_2 = QSlider(Qt.Horizontal)
        slider_2.setRange(0, 100)
        slider_2.setValue(50)
        placeholder_layout.addWidget(QLabel("Paper bias: 50"))
        placeholder_layout.addWidget(slider_2)

        layout.addWidget(placeholder_group)
        return page

    def _create_results_tab(self):
        """Tab 5: run the analysis and display a readable report."""
        page = QWidget()
        layout = QVBoxLayout(page)

        self.status_label = QLabel("Load B1, B2 and A to compare images.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 15px; margin: 8px;")
        layout.addWidget(self.status_label)

        analyze_button = QPushButton("Analyse")
        analyze_button.clicked.connect(self.analyze_images)
        layout.addWidget(analyze_button)

        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setPlaceholderText("Analysis summary will appear here.")
        layout.addWidget(self.report)
        return page

    def analyze_images(self):
        """Run the comparison for the three loaded images and report the result."""
        if self.image_paths["B1"] is None or self.image_paths["B2"] is None or self.image_paths["A"] is None:
            QMessageBox.warning(self, "Missing input", "Load B1, B2, and A before running the analysis.")
            return

        try:
            b1 = load_image(self.image_paths["B1"])
            b2 = load_image(self.image_paths["B2"])
            a = load_image(self.image_paths["A"])
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
            self.status_label.setText("Analysis complete. The report compares B1 against both B2 and A.")
        except Exception as exc:
            QMessageBox.critical(self, "Analysis failed", f"Unable to analyze the images:\n{exc}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
