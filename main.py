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
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QColorDialog,
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
    QSizePolicy,
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
        self.ink_mask_arrays = {"B1": None, "B2": None, "A": None}
        self.paper_mask_arrays = {"B1": None, "B2": None, "A": None}
        self.mask_arrays = self.ink_mask_arrays
        self.mask_border_arrays = {"B1": None, "B2": None, "A": None}
        self.mask_reference_colours = {
            "B1": {"paper": QColor("white"), "ink": QColor("black")},
            "B2": {"paper": QColor("white"), "ink": QColor("black")},
            "A": {"paper": QColor("white"), "ink": QColor("black")},
        }
        self.current_alignment_view = "B1"
        self.alignment_display_mode = "original"
        self.mask_overlay_colour = QColor("blue")
        self.mask_mode = "luminance"

        self.tab_widget = QTabWidget()
        self.setCentralWidget(self.tab_widget)

        self.load_tab = self._create_load_tab()
        self.mask_tab = self._create_mask_tab()
        self.check_alignment_tab = self._create_check_alignment_tab()
        self.analysis_options_tab = self._create_analysis_options_tab()
        self.results_tab = self._create_results_tab()

        self.tab_widget.addTab(self.load_tab, "1) Load images")
        self.tab_widget.addTab(self.mask_tab, "2) Mask")
        self.tab_widget.addTab(self.check_alignment_tab, "3) Check alignment")
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

    def _contrast_text_colour(self, colour: QColor) -> str:
        """Return black or white based on the luminance of a QColor."""
        red, green, blue = colour.getRgb()[:3]
        luminance = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255.0
        return "#000000" if luminance > 0.55 else "#ffffff"

    def _apply_button_colour(self, button: QPushButton, colour: QColor):
        """Set a button background colour while keeping the text readable."""
        button.setStyleSheet(
            f"background-color: {colour.name()}; color: {self._contrast_text_colour(colour)}; "
            "border: 1px solid #555;"
        )

    def _update_overlay_button_style(self):
        """Display the current overlay colour on the check-alignment button."""
        if hasattr(self, "overlay_button"):
            self._apply_button_colour(self.overlay_button, self.mask_overlay_colour)
            self.overlay_button.setText(f"Overlay colour: {self.mask_overlay_colour.name()}")

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

        if hasattr(self, "mask_original_labels") and key in self.mask_original_labels:
            self._preview_from_array(self.mask_original_labels[key], image)

        if self.current_alignment_view == key:
            self._display_alignment_image()

    def _create_load_tab(self):
        """Tab 1: load the three images and show their file paths."""
        page = QWidget()
        page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop)

        input_group = QGroupBox("Image type")
        input_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        input_group.setMaximumHeight(120)
        input_group.setMaximumWidth(260)
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
            path_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            path_label.setAlignment(Qt.AlignTop)
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

        mode = self.alignment_display_mode
        if mode == "original":
            self._preview_from_array(self.alignment_preview, image)
            return

        if mode == "mask":
            if not self.use_mask_checkbox.isChecked():
                self.alignment_preview.clear()
                self.alignment_preview.setText("No mask found")
                return
            mask = self.ink_mask_arrays.get(key)
            if mask is None:
                self.alignment_preview.clear()
                self.alignment_preview.setText("No mask found")
                return
            self._display_mask_preview(self.alignment_preview, mask, self.mask_border_arrays.get(key))
            return

        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        array = np.clip(array, 0.0, 1.0)

        paper_mask = self.paper_mask_arrays.get(key)
        ink_mask = self.ink_mask_arrays.get(key)
        if not self.use_mask_checkbox.isChecked() or paper_mask is None or ink_mask is None:
            self._preview_from_array(self.alignment_preview, image)
            return

        paper_bool = np.asarray(paper_mask, dtype=bool)
        ink_bool = np.asarray(ink_mask, dtype=bool)
        overlay = np.array(self.mask_overlay_colour.getRgb()[:3], dtype=np.float32) / 255.0
        displayed = array.copy()

        if mode in {"paper_mask", "masked_paper"}:
            displayed[~paper_bool] = overlay
        elif mode in {"ink_mask", "masked_ink"}:
            displayed[~ink_bool] = overlay
        else:
            displayed[~paper_bool] = overlay

        rgb = (displayed * 255.0).astype("uint8")
        rgb = np.ascontiguousarray(rgb)
        h, w, ch = rgb.shape
        qimage = QImage(rgb.data, w, h, w * ch, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())
        self.alignment_preview.setPixmap(
            pixmap.scaled(self.alignment_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

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
        """Tab 3: visually inspect the alignment by switching the view between B1/B2/A."""
        page = QWidget()
        page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop)

        self.alignment_checkbox = QCheckBox("Images are aligned")
        self.alignment_checkbox.setChecked(False)
        layout.addWidget(self.alignment_checkbox)

        view_group = QGroupBox("Alignment view")
        view_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        view_layout = QHBoxLayout(view_group)
        self.alignment_mode_group = QButtonGroup(page)
        self.original_radio = QRadioButton("Original")
        self.mask_radio = QRadioButton("Mask")
        self.masked_paper_radio = QRadioButton("Paper mask")
        self.masked_ink_radio = QRadioButton("Ink mask")
        self.original_radio.setChecked(True)
        self.alignment_mode_group.addButton(self.original_radio)
        self.alignment_mode_group.addButton(self.mask_radio)
        self.alignment_mode_group.addButton(self.masked_paper_radio)
        self.alignment_mode_group.addButton(self.masked_ink_radio)
        self.original_radio.toggled.connect(lambda checked: self._set_alignment_mode("original") if checked else None)
        self.mask_radio.toggled.connect(lambda checked: self._set_alignment_mode("mask") if checked else None)
        self.masked_paper_radio.toggled.connect(lambda checked: self._set_alignment_mode("paper_mask") if checked else None)
        self.masked_ink_radio.toggled.connect(lambda checked: self._set_alignment_mode("ink_mask") if checked else None)
        for radio in [self.original_radio, self.mask_radio, self.masked_paper_radio, self.masked_ink_radio]:
            view_layout.addWidget(radio)
        layout.addWidget(view_group)

        self.overlay_button = QPushButton(f"Overlay colour: {self.mask_overlay_colour.name()}")
        self.overlay_button.setFixedWidth(220)
        self.overlay_button.clicked.connect(self._choose_mask_overlay_colour)
        self._update_overlay_button_style()
        overlay_row = QHBoxLayout()
        overlay_row.setAlignment(Qt.AlignLeft)
        overlay_row.addWidget(self.overlay_button)
        overlay_row.addStretch(1)
        layout.addLayout(overlay_row)

        button_row = QHBoxLayout()
        button_row.setAlignment(Qt.AlignLeft)
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
        self.alignment_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.alignment_preview.setStyleSheet("border: 1px solid #888; background: #262626;")
        self.alignment_preview.clicked.connect(self._cycle_alignment_image)
        layout.addWidget(self.alignment_preview)

        self._update_alignment_buttons()
        self._refresh_alignment_preview()
        return page

    def _set_alignment_view(self, key: str):
        self.current_alignment_view = key
        self._update_alignment_buttons()
        self._display_alignment_image()

    def _set_alignment_mode(self, mode: str):
        self.alignment_display_mode = mode
        self._display_alignment_image()

    def _refresh_alignment_preview(self):
        self._display_alignment_image()

    def _choose_mask_overlay_colour(self):
        colour = QColorDialog.getColor(self.mask_overlay_colour, self, "Select masked area colour")
        if colour.isValid():
            self.mask_overlay_colour = colour
            self._update_overlay_button_style()
            self._display_alignment_image()

    def _set_mask_mode(self, mode: str):
        self.mask_mode = mode
        use_mask_enabled = self.use_mask_checkbox.isChecked() if hasattr(self, "use_mask_checkbox") else True
        reference_mode_enabled = use_mask_enabled and mode == "reference_colour"
        luminance_mode_enabled = use_mask_enabled and mode == "luminance"

        if hasattr(self, "reference_weight_slider"):
            self.reference_weight_slider.setEnabled(reference_mode_enabled)
            self.reference_weight_label.setEnabled(reference_mode_enabled)
        if hasattr(self, "mask_threshold_slider"):
            self.mask_threshold_slider.setEnabled(luminance_mode_enabled)
            self.mask_threshold_label.setEnabled(luminance_mode_enabled)
        if hasattr(self, "change_all_paper_button"):
            self.change_all_paper_button.setEnabled(reference_mode_enabled)
        if hasattr(self, "change_all_ink_button"):
            self.change_all_ink_button.setEnabled(reference_mode_enabled)
        if hasattr(self, "paper_colour_buttons"):
            for key in ["B1", "B2", "A"]:
                self.paper_colour_buttons[key].setEnabled(reference_mode_enabled)
                self.ink_colour_buttons[key].setEnabled(reference_mode_enabled)
                if reference_mode_enabled:
                    self._apply_button_colour(self.paper_colour_buttons[key], self.mask_reference_colours[key]["paper"])
                    self._apply_button_colour(self.ink_colour_buttons[key], self.mask_reference_colours[key]["ink"])
                else:
                    self.paper_colour_buttons[key].setStyleSheet("background-color: #4d4d4d; color: #7a7a7a; border: 1px solid #666;")
                    self.ink_colour_buttons[key].setStyleSheet("background-color: #4d4d4d; color: #7a7a7a; border: 1px solid #666;")
        if hasattr(self, "generate_all_masks_button"):
            self.generate_all_masks_button.setEnabled(use_mask_enabled)
        if hasattr(self, "mask_generate_buttons"):
            for button in self.mask_generate_buttons.values():
                button.setEnabled(use_mask_enabled)
        if hasattr(self, "mask_remove_buttons"):
            for button in self.mask_remove_buttons.values():
                button.setEnabled(use_mask_enabled)
        if hasattr(self, "luminance_radio"):
            self.luminance_radio.setEnabled(use_mask_enabled)
        if hasattr(self, "colour_radio"):
            self.colour_radio.setEnabled(use_mask_enabled)
        if hasattr(self, "alignment_preview"):
            self._display_alignment_image()

    def _set_mask_reference_colour(self, key: str, region: str):
        colour = QColorDialog.getColor(self.mask_reference_colours[key][region], self, f"Select {region} colour for {key}")
        if not colour.isValid():
            return

        self.mask_reference_colours[key][region] = colour
        self._update_reference_colour_buttons()

    def _change_all_reference_colours(self, region: str):
        representative = self.mask_reference_colours["B1"][region]
        colour = QColorDialog.getColor(representative, self, f"Select all {region} colours")
        if not colour.isValid():
            return

        for key in ["B1", "B2", "A"]:
            self.mask_reference_colours[key][region] = colour

        self._update_reference_colour_buttons()

    def _update_reference_colour_buttons(self):
        for key in ["B1", "B2", "A"]:
            paper_colour = self.mask_reference_colours[key]["paper"]
            ink_colour = self.mask_reference_colours[key]["ink"]
            self._apply_button_colour(self.paper_colour_buttons[key], paper_colour)
            self._apply_button_colour(self.ink_colour_buttons[key], ink_colour)
            self.paper_colour_buttons[key].setText(f"Paper: {paper_colour.name()}")
            self.ink_colour_buttons[key].setText(f"Ink: {ink_colour.name()}")

        if hasattr(self, "change_all_paper_button"):
            self.change_all_paper_button.setStyleSheet("")
            self.change_all_paper_button.setText("Set all paper colours")
        if hasattr(self, "change_all_ink_button"):
            self.change_all_ink_button.setStyleSheet("")
            self.change_all_ink_button.setText("Set all ink colours")

    def _generate_mask_for_key(self, key: str):
        """Create separate paper and ink masks and keep the red border out of both."""
        image = self.loaded_arrays.get(key)
        if image is None:
            QMessageBox.warning(self, "Missing image", f"Load {key} before generating a mask.")
            return

        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 3:
            rgb_array = array.copy()
            if rgb_array.max() > 1.0:
                rgb_array = rgb_array / 255.0
        else:
            rgb_array = np.repeat(array[..., None], 3, axis=-1)
            if rgb_array.max() > 1.0:
                rgb_array = rgb_array / 255.0

        if self.mask_mode == "luminance":
            brightness = np.mean(rgb_array, axis=-1)
            threshold = self.mask_threshold_slider.value() / 255.0
            ink_mask = brightness <= threshold
        else:
            paper_colour = np.array(self.mask_reference_colours[key]["paper"].getRgb()[:3], dtype=np.float32) / 255.0
            ink_colour = np.array(self.mask_reference_colours[key]["ink"].getRgb()[:3], dtype=np.float32) / 255.0
            paper_distance = np.linalg.norm(rgb_array - paper_colour, axis=-1)
            ink_distance = np.linalg.norm(rgb_array - ink_colour, axis=-1)
            max_distance = np.sqrt(3.0)
            paper_weight = self.reference_weight_slider.value() / 100.0
            adjusted_paper_distance = paper_weight * paper_distance + (1.0 - paper_weight) * max_distance
            adjusted_ink_distance = (1.0 - paper_weight) * ink_distance + paper_weight * max_distance
            ink_mask = adjusted_paper_distance > adjusted_ink_distance

        paper_mask = ~ink_mask
        border_pixels = self.mask_border_slider.value()
        if border_pixels > 0:
            border_mask = ink_mask ^ binary_erosion(ink_mask, iterations=border_pixels)
        else:
            border_mask = np.zeros_like(ink_mask, dtype=bool)

        ink_mask = ink_mask & ~border_mask
        paper_mask = paper_mask & ~border_mask

        self.ink_mask_arrays[key] = ink_mask
        self.paper_mask_arrays[key] = paper_mask
        self.mask_arrays[key] = ink_mask
        self.mask_border_arrays[key] = border_mask

        if self.ink_mask_arrays.get(key) is not None:
            self._display_mask_preview(
                self.mask_preview_labels[key],
                self.ink_mask_arrays[key],
                self.mask_border_arrays.get(key),
            )

        self._display_alignment_image()

    def _generate_mask_for_all(self):
        """Generate masks for all images currently loaded into the app."""
        for key in ["B1", "B2", "A"]:
            if self.loaded_arrays.get(key) is not None:
                self._generate_mask_for_key(key)

    def _remove_mask_for_key(self, key: str):
        """Clear the saved mask for a single image, leaving the original image visible."""
        self.ink_mask_arrays[key] = None
        self.paper_mask_arrays[key] = None
        self.mask_arrays[key] = None
        self.mask_border_arrays[key] = None

        if key in self.mask_preview_labels:
            self.mask_preview_labels[key].clear()
            self.mask_preview_labels[key].setText(f"Mask {key}")

        self._display_alignment_image()

    def _update_mask_generation_button_text(self):
        if not hasattr(self, "mask_generate_buttons"):
            return

        for key, button in self.mask_generate_buttons.items():
            button.setText(f"Generate mask for {key}")

    def _set_mask_controls_enabled(self, enabled: bool):
        """Enable or disable the whole mask section without overriding the active mode rules."""
        if hasattr(self, "use_mask_checkbox"):
            self.use_mask_checkbox.setChecked(enabled)

        if hasattr(self, "mask_mode_group"):
            for button in self.mask_mode_group.buttons():
                button.setEnabled(enabled)

        if hasattr(self, "mask_threshold_slider"):
            self.mask_threshold_slider.setEnabled(enabled and self.mask_mode == "luminance")
            self.mask_threshold_label.setEnabled(enabled and self.mask_mode == "luminance")
        if hasattr(self, "reference_weight_slider"):
            self.reference_weight_slider.setEnabled(enabled and self.mask_mode == "reference_colour")
            self.reference_weight_label.setEnabled(enabled and self.mask_mode == "reference_colour")
        if hasattr(self, "mask_border_slider"):
            self.mask_border_slider.setEnabled(enabled)
            self.mask_border_label.setEnabled(enabled)

        if hasattr(self, "paper_colour_buttons"):
            for key, button in self.paper_colour_buttons.items():
                should_enable = enabled and self.mask_mode == "reference_colour"
                button.setEnabled(should_enable)
                if should_enable:
                    self._apply_button_colour(button, self.mask_reference_colours[key]["paper"])
                else:
                    button.setStyleSheet("background-color: #4d4d4d; color: #7a7a7a; border: 1px solid #666;")

        if hasattr(self, "ink_colour_buttons"):
            for key, button in self.ink_colour_buttons.items():
                should_enable = enabled and self.mask_mode == "reference_colour"
                button.setEnabled(should_enable)
                if should_enable:
                    self._apply_button_colour(button, self.mask_reference_colours[key]["ink"])
                else:
                    button.setStyleSheet("background-color: #4d4d4d; color: #7a7a7a; border: 1px solid #666;")

        if hasattr(self, "change_all_paper_button"):
            self.change_all_paper_button.setEnabled(enabled and self.mask_mode == "reference_colour")
        if hasattr(self, "change_all_ink_button"):
            self.change_all_ink_button.setEnabled(enabled and self.mask_mode == "reference_colour")
        if hasattr(self, "generate_all_masks_button"):
            self.generate_all_masks_button.setEnabled(enabled)
        if hasattr(self, "mask_generate_buttons"):
            for button in self.mask_generate_buttons.values():
                button.setEnabled(enabled)
        if hasattr(self, "mask_remove_buttons"):
            for button in self.mask_remove_buttons.values():
                button.setEnabled(enabled)
        if hasattr(self, "mask_preview_labels"):
            for label in self.mask_preview_labels.values():
                label.setEnabled(enabled)
        if hasattr(self, "mask_original_labels"):
            for label in self.mask_original_labels.values():
                label.setEnabled(enabled)

        if hasattr(self, "alignment_preview"):
            self._display_alignment_image()

    def _create_mask_tab(self):
        """Tab 3: prepare and preview masks used for the paper and ink analysis."""
        page = QWidget()
        page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop)

        self.use_mask_checkbox = QCheckBox("Use mask")
        self.use_mask_checkbox.setChecked(True)
        self.use_mask_checkbox.toggled.connect(lambda checked: self._set_mask_controls_enabled(checked))
        layout.addWidget(self.use_mask_checkbox)

        settings_group = QGroupBox("Mask settings")
        settings_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        settings_layout = QVBoxLayout(settings_group)

        mode_group = QGroupBox("Mask generation")
        mode_group.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        mode_group.setMaximumWidth(280)
        mode_layout = QHBoxLayout(mode_group)
        self.mask_mode_group = QButtonGroup(page)
        self.luminance_radio = QRadioButton("Luminance")
        self.colour_radio = QRadioButton("Reference colour")
        self.luminance_radio.setChecked(True)
        self.mask_mode_group.addButton(self.luminance_radio)
        self.mask_mode_group.addButton(self.colour_radio)
        self.luminance_radio.toggled.connect(lambda checked: self._set_mask_mode("luminance") if checked else None)
        self.colour_radio.toggled.connect(lambda checked: self._set_mask_mode("reference_colour") if checked else None)
        mode_layout.addWidget(self.luminance_radio)
        mode_layout.addWidget(self.colour_radio)
        settings_layout.addWidget(mode_group)

        self.mask_threshold_slider = QSlider(Qt.Horizontal)
        self.mask_threshold_slider.setRange(0, 255)
        self.mask_threshold_slider.setValue(140)
        self.mask_threshold_label = QLabel("Luminance threshold: 140")
        self.mask_threshold_slider.setEnabled(True)
        self.mask_threshold_slider.valueChanged.connect(
            lambda value: self.mask_threshold_label.setText(f"Luminance threshold: {value}")
        )
        settings_layout.addWidget(self.mask_threshold_label)
        settings_layout.addWidget(self.mask_threshold_slider)

        self.reference_weight_slider = QSlider(Qt.Horizontal)
        self.reference_weight_slider.setRange(0, 100)
        self.reference_weight_slider.setValue(50)
        self.reference_weight_slider.setEnabled(False)
        self.reference_weight_label = QLabel("Reference weight: 0.50")
        self.reference_weight_slider.valueChanged.connect(
            lambda value: self.reference_weight_label.setText(f"Reference weight: {value / 100.0:.2f}")
        )
        settings_layout.addWidget(self.reference_weight_label)
        settings_layout.addWidget(self.reference_weight_slider)

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
        self.mask_original_labels = {}
        self.paper_colour_buttons = {}
        self.ink_colour_buttons = {}
        self.mask_generate_buttons = {}
        self.mask_remove_buttons = {}
        self.mask_border_arrays = {"B1": None, "B2": None, "A": None}
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignTop)
        for col, key in enumerate(["B1", "B2", "A"]):
            card = QWidget()
            card_layout = QVBoxLayout(card)

            original_preview = QLabel(f"Original {key}")
            original_preview.setAlignment(Qt.AlignCenter)
            original_preview.setMinimumSize(240, 140)
            original_preview.setStyleSheet("border: 1px solid #888; background: #262626;")
            card_layout.addWidget(original_preview)

            mask_preview = QLabel(f"Mask {key}")
            mask_preview.setAlignment(Qt.AlignCenter)
            mask_preview.setMinimumSize(240, 140)
            mask_preview.setStyleSheet("border: 1px solid #888; background: #262626;")
            card_layout.addWidget(mask_preview)

            colour_row = QHBoxLayout()
            paper_button = QPushButton("Paper colour")
            paper_button.setMinimumWidth(120)
            paper_button.clicked.connect(lambda checked, value=key: self._set_mask_reference_colour(value, "paper"))
            ink_button = QPushButton("Ink colour")
            ink_button.setMinimumWidth(120)
            ink_button.clicked.connect(lambda checked, value=key: self._set_mask_reference_colour(value, "ink"))
            colour_row.addWidget(paper_button)
            colour_row.addWidget(ink_button)
            card_layout.addLayout(colour_row)

            button = QPushButton(f"Generate mask for {key}")
            button.setMinimumWidth(180)
            button.clicked.connect(lambda checked, value=key: self._generate_mask_for_key(value))
            card_layout.addWidget(button)

            remove_button = QPushButton(f"Remove mask for {key}")
            remove_button.setMinimumWidth(180)
            remove_button.clicked.connect(lambda checked, value=key: self._remove_mask_for_key(value))
            card_layout.addWidget(remove_button)

            self.mask_original_labels[key] = original_preview
            self.mask_preview_labels[key] = mask_preview
            self.paper_colour_buttons[key] = paper_button
            self.ink_colour_buttons[key] = ink_button
            self.mask_generate_buttons[key] = button
            self.mask_remove_buttons[key] = remove_button
            grid.addWidget(card, 0, col)

        self.change_all_paper_button = QPushButton("Set all paper colours")
        self.change_all_paper_button.clicked.connect(lambda: self._change_all_reference_colours("paper"))
        self.change_all_ink_button = QPushButton("Set all ink colours")
        self.change_all_ink_button.clicked.connect(lambda: self._change_all_reference_colours("ink"))
        self.change_all_paper_button.setFixedWidth(220)
        self.change_all_ink_button.setFixedWidth(220)

        self.generate_all_masks_button = QPushButton("Generate masks for all")
        self.generate_all_masks_button.clicked.connect(self._generate_mask_for_all)
        self.generate_all_masks_button.setFixedWidth(220)

        settings_layout.addWidget(self.change_all_paper_button)
        settings_layout.addWidget(self.change_all_ink_button)
        settings_layout.addWidget(self.generate_all_masks_button)

        layout.addLayout(grid)
        self._update_reference_colour_buttons()
        self._update_mask_generation_button_text()
        self._set_mask_controls_enabled(self.use_mask_checkbox.isChecked())
        self._set_mask_mode("luminance")
        return page

    def _create_analysis_options_tab(self):
        """Tab 4: future analysis settings kept deliberately blank and simple."""
        page = QWidget()
        page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop)

        placeholder_group = QGroupBox("Future controls")
        placeholder_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        placeholder_group.setMaximumHeight(220)
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
        page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop)

        self.status_label = QLabel("Load B1, B2 and A to compare images.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 15px; margin: 8px;")
        layout.addWidget(self.status_label)

        analyze_button = QPushButton("Analyse")
        analyze_button.setFixedWidth(170)
        analyze_button.clicked.connect(self.analyze_images)
        button_row = QHBoxLayout()
        button_row.setAlignment(Qt.AlignLeft)
        button_row.addWidget(analyze_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.report = QTextEdit()
        self.report.setReadOnly(True)
        self.report.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
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
