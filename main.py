"""Desktop app for paper conservation analysis.

The workflow is intentionally simple: the images are assumed to have been aligned
and cropped before they reach this application, and the app focuses on the
scientific comparison and report generation.
"""

import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion

# Add the source tree to the import path so the app can import the package.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
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

from paperdiff.image_state import analysis_options, image_data_colour
from paperdiff.loading import load_image
from paperdiff.stats import verdict_for_region
from paperdiff.workflow import describe_region_change, compare_images


class ZoomableLabel(QLabel):
    """A QLabel that keeps a base pixmap and allows zooming via the mouse wheel."""

    clicked = Signal()

    def __init__(self, text=""):
        super().__init__(text)
        self._base_pixmap = QPixmap()
        self._zoom_factor = 1.0
        self._auto_fit_zoom = True
        self._fixed_viewport = QSize(320, 240)
        self._pan_x = 0.0
        self._pan_y = 0.0
        self.setMinimumSize(120, 80)

    def sizeHint(self):
        return self._fixed_viewport if self._fixed_viewport.isValid() else QSize(320, 240)

    def minimumSizeHint(self):
        return QSize(120, 80)

    def _clamp_pan(self, viewport: QSize, scaled_width: int, scaled_height: int, pan_x: float | None = None, pan_y: float | None = None):
        if pan_x is None:
            pan_x = self._pan_x
        if pan_y is None:
            pan_y = self._pan_y

        if scaled_width <= viewport.width():
            pan_x = (viewport.width() - scaled_width) / 2.0
        else:
            pan_x = min(0.0, max(viewport.width() - scaled_width, pan_x))

        if scaled_height <= viewport.height():
            pan_y = (viewport.height() - scaled_height) / 2.0
        else:
            pan_y = min(0.0, max(viewport.height() - scaled_height, pan_y))

        return pan_x, pan_y

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fixed_viewport = self.size()
        if self._base_pixmap is not None and not self._base_pixmap.isNull():
            if self._auto_fit_zoom:
                self._apply_fit_zoom()
            self._refresh_zoomed_pixmap()

    def setPixmap(self, pixmap, preserve_zoom: bool = False):
        self._base_pixmap = pixmap if pixmap is not None else QPixmap()
        if preserve_zoom:
            self._auto_fit_zoom = False
            self._refresh_zoomed_pixmap()
            return

        self._auto_fit_zoom = True
        self._apply_fit_zoom()
        self._refresh_zoomed_pixmap()

    def setFixedViewport(self, width: int, height: int):
        self._fixed_viewport = QSize(width, height)
        self._apply_fit_zoom()
        self._refresh_zoomed_pixmap()

    def _fit_zoom_for_viewport(self, viewport: QSize | None = None) -> float:
        """Return the maximum zoom-out value that keeps the entire image inside the current viewport."""
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return 1.0

        if viewport is None:
            viewport = self._fixed_viewport if self._fixed_viewport.width() > 0 and self._fixed_viewport.height() > 0 else self.size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return 1.0

        source_size = self._base_pixmap.size()
        if source_size.width() <= 0 or source_size.height() <= 0:
            return 1.0

        fit_factor = min(viewport.width() / source_size.width(), viewport.height() / source_size.height())
        return max(0.05, min(12.0, fit_factor))

    def _apply_fit_zoom(self):
        """Default to the maximum zoom-out state so the whole image stays visible."""
        fit_zoom = self._fit_zoom_for_viewport()
        self._zoom_factor = max(0.05, min(12.0, fit_zoom))
        self._pan_x = 0.0
        self._pan_y = 0.0

    def _render_centered_pixmap(self, source_pixmap: QPixmap, viewport: QSize) -> QPixmap:
        """Paint the image inside the viewport and keep the current pan/zoom anchored."""
        canvas = QPixmap(viewport)
        canvas.fill(self.palette().color(self.backgroundRole()))

        if source_pixmap.isNull() or viewport.width() <= 0 or viewport.height() <= 0:
            return canvas

        scaled = source_pixmap.scaled(
            QSize(max(1, int(source_pixmap.width() * self._zoom_factor)), max(1, int(source_pixmap.height() * self._zoom_factor))),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )

        if scaled.width() <= 0 or scaled.height() <= 0:
            return canvas

        self._pan_x, self._pan_y = self._clamp_pan(viewport, scaled.width(), scaled.height())
        painter = QPainter(canvas)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(int(round(self._pan_x)), int(round(self._pan_y)), scaled)
        painter.end()
        return canvas

    def _refresh_zoomed_pixmap(self):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            super().setPixmap(QPixmap())
            return

        viewport = self._fixed_viewport if self._fixed_viewport.width() > 0 and self._fixed_viewport.height() > 0 else self.size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            viewport = self._base_pixmap.size()

        minimum_zoom = self._fit_zoom_for_viewport(viewport)
        self._zoom_factor = max(minimum_zoom, min(12.0, self._zoom_factor))
        self._pan_x, self._pan_y = self._clamp_pan(viewport, max(1, int(self._base_pixmap.width() * self._zoom_factor)), max(1, int(self._base_pixmap.height() * self._zoom_factor)))
        super().setPixmap(self._render_centered_pixmap(self._base_pixmap, viewport))

    def wheelEvent(self, event):
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return

        self._auto_fit_zoom = False
        viewport = self.size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return

        old_zoom = self._zoom_factor
        delta = event.angleDelta().y()
        fit_zoom = self._fit_zoom_for_viewport()
        if delta > 0:
            new_zoom = min(12.0, max(fit_zoom, old_zoom * 1.15))
        elif delta < 0:
            new_zoom = max(fit_zoom, old_zoom * 0.85)
        else:
            return

        old_scaled_width = max(1, int(self._base_pixmap.width() * old_zoom))
        old_scaled_height = max(1, int(self._base_pixmap.height() * old_zoom))

        cursor_x = event.position().x()
        cursor_y = event.position().y()

        anchor_x = cursor_x - self._pan_x
        anchor_y = cursor_y - self._pan_y
        anchor_u = max(0.0, min(1.0, anchor_x / old_scaled_width))
        anchor_v = max(0.0, min(1.0, anchor_y / old_scaled_height))

        new_scaled_width = max(1, int(self._base_pixmap.width() * new_zoom))
        new_scaled_height = max(1, int(self._base_pixmap.height() * new_zoom))
        new_pan_x = cursor_x - (anchor_u * new_scaled_width)
        new_pan_y = cursor_y - (anchor_v * new_scaled_height)

        self._zoom_factor = new_zoom
        self._pan_x, self._pan_y = self._clamp_pan(viewport, new_scaled_width, new_scaled_height, new_pan_x, new_pan_y)
        self._refresh_zoomed_pixmap()
        event.accept()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

    def clear(self):
        self._base_pixmap = QPixmap()
        self._zoom_factor = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        super().clear()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)


class ClickableLabel(ZoomableLabel):
    """A zoomable label that emits a click signal so the alignment viewer can cycle images."""

    pass


class MainWindow(QMainWindow):
    """The main window containing the five-step analysis workflow."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Paper Conservation Analysis")
        self.setMinimumSize(0, 0)
        self.resize(1200, 800)

        # The scientific workflow assumes the images are already aligned and cropped
        # before they are opened in this app.
        self.images = {key: image_data_colour(key) for key in ["B1", "B2", "A"]}
        self.mask_reference_colours = {key: self.images[key].reference_colours for key in ["B1", "B2", "A"]}
        self.analysis_options = analysis_options()
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

        self._refresh_preview_box_sizes()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_preview_box_sizes()

    def _refresh_preview_box_sizes(self):
        """Keep the small preview cards and the alignment viewer scaled by the current window size, without forcing fixed internal viewport dimensions."""
        available_width = max(240, self.width() - 180)
        thumb_width = max(180, min(360, int(available_width / 3)))
        available_height = max(180, self.height() - 300)
        thumb_height = max(90, min(int(thumb_width * 0.55), available_height))

        if hasattr(self, "load_preview_labels"):
            for label in self.load_preview_labels.values():
                label.setMinimumSize(80, 60)

        if hasattr(self, "alignment_preview"):
            self.alignment_preview.setMinimumSize(200, 120)

        if hasattr(self, "mask_original_labels"):
            for label in self.mask_original_labels.values():
                label.setMinimumSize(80, 60)
                label.setFixedHeight(thumb_height)
                if hasattr(label, "_apply_fit_zoom"):
                    label._apply_fit_zoom()
                    label._refresh_zoomed_pixmap()
            for label in self.mask_preview_labels.values():
                label.setMinimumSize(80, 60)
                label.setFixedHeight(thumb_height)

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

        if isinstance(label, ZoomableLabel):
            label.setPixmap(pixmap, preserve_zoom=True)
            return

        viewport = label.size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            label.setPixmap(pixmap)
            return

        fit_pixmap = QPixmap(viewport)
        fit_pixmap.fill(label.palette().color(label.backgroundRole()))
        scaled = pixmap.scaled(viewport, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        offset_x = max(0, (viewport.width() - scaled.width()) // 2)
        offset_y = max(0, (viewport.height() - scaled.height()) // 2)
        painter = QPainter(fit_pixmap)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(offset_x, offset_y, scaled)
        painter.end()
        label.setPixmap(fit_pixmap)

    def _as_bool_mask(self, mask):
        """Convert 2D or 3-channel mask arrays to a 2D bool mask for UI logic."""
        if mask is None:
            return None
        array = np.asarray(mask, dtype=np.float32)
        if array.ndim == 3:
            if array.shape[-1] == 1:
                array = array[..., 0]
            else:
                array = array[..., 0]
        return array.astype(bool)

    def _display_mask_preview(self, label: QLabel, mask_array=None, border_mask=None, paper_mask=None, ink_mask=None):
        """Render the classification as white paper, black ink, and red for pixels in neither mask."""
        if paper_mask is not None and ink_mask is not None:
            paper = self._as_bool_mask(paper_mask)
            ink = self._as_bool_mask(ink_mask)
            h, w = paper.shape
            rendered = np.zeros((h, w, 3), dtype=np.uint8)
            rendered[paper] = [255, 255, 255]
            rendered[ink] = [0, 0, 0]
            rendered[~paper & ~ink] = [255, 0, 0]
        else:
            if mask_array is None:
                label.setText("No mask")
                label.setPixmap(QPixmap())
                return

            mask = self._as_bool_mask(mask_array)
            if border_mask is None:
                border_mask = mask ^ binary_erosion(mask, iterations=1)
            else:
                border_mask = self._as_bool_mask(border_mask)

            h, w = mask.shape
            rendered = np.zeros((h, w, 3), dtype=np.uint8)
            rendered[mask] = [255, 255, 255]
            rendered[border_mask] = [255, 0, 0]

        qimage = QImage(rendered.data, w, h, w * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())

        if isinstance(label, ZoomableLabel):
            label.setPixmap(pixmap, preserve_zoom=True)
            return

        viewport = label.size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            label.setPixmap(pixmap)
            return

        fit_pixmap = QPixmap(viewport)
        fit_pixmap.fill(label.palette().color(label.backgroundRole()))
        scaled = pixmap.scaled(viewport, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        offset_x = max(0, (viewport.width() - scaled.width()) // 2)
        offset_y = max(0, (viewport.height() - scaled.height()) // 2)
        painter = QPainter(fit_pixmap)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawPixmap(offset_x, offset_y, scaled)
        painter.end()
        label.setPixmap(fit_pixmap)

    def _format_file_size(self, size_bytes: int) -> str:
        """Convert byte counts into a readable value suitable for display in the UI."""
        units = ["B", "KB", "MB", "GB"]
        value = float(size_bytes)
        unit_index = 0
        while value >= 1024 and unit_index < len(units) - 1:
            value /= 1024.0
            unit_index += 1

        if unit_index == 0:
            return f"{int(value)} {units[unit_index]}"
        return f"{value:.1f} {units[unit_index]}"

    def _image_file_details(self, path: str) -> str:
        """Return a short metadata summary for an image file."""
        if not path:
            return "No image details"

        try:
            file_path = Path(path)
            file_size = file_path.stat().st_size
            with Image.open(file_path) as image:
                width, height = image.size
                bits = getattr(image, "bits", "unknown")
                format_name = (image.format or file_path.suffix.lstrip(".") or "unknown").lower()
                return f"{width} × {height}px • {bits}-bit • {format_name} • {self._format_file_size(file_size)}"
        except Exception:
            suffix = Path(path).suffix.lstrip(".").lower() or "unknown"
            size = Path(path).stat().st_size if Path(path).exists() else 0
            return f"Unknown size • {suffix} • {self._format_file_size(size)}"

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
            self.overlay_button.setText(f" Mask overlay colour: {self.mask_overlay_colour.name()} ")

    def _ensure_image_loaded_for_key(self, key: str):
        """Restore an image object from disk if the loaded state has been lost but the file path still exists."""
        image_data = self.images[key]
        if image_data.pixels.get("rgb") is not None:
            return True

        path = image_data.file_path
        if not path or not Path(path).exists():
            return False

        try:
            image_data.load_from_path(path)
            return True
        except Exception:
            return False

    def _refresh_loaded_image_previews_for_key(self, key: str):
        """Keep the load, mask, and alignment previews in sync for the selected image key."""
        if not self._ensure_image_loaded_for_key(key):
            return

        image = self.images[key].get_pixels("rgb")

        if hasattr(self, "load_preview_labels") and key in self.load_preview_labels:
            self._preview_from_array(self.load_preview_labels[key], image)

        if hasattr(self, "mask_original_labels") and key in self.mask_original_labels:
            self._preview_from_array(self.mask_original_labels[key], image)

        if hasattr(self, "alignment_preview") and self.current_alignment_view == key:
            self._display_alignment_image()

    def _open_image_for_key(self, key: str):
        """Open a single image and store it in the image object for the given key."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {key} image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return

        image_data = self.images[key]
        image_data.load_from_path(path)

        preview_label = self.load_preview_labels[key]
        self._preview_from_array(preview_label, image_data.get_pixels("rgb"))
        self.path_labels[key].setText(path)
        self.image_details_labels[key].setText(self._image_file_details(path))

        self._refresh_loaded_image_previews_for_key(key)

        if hasattr(self, "_update_alignment_mode_availability"):
            self._update_alignment_mode_availability()

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
        self.image_details_labels = {}
        self.image_key_labels = {"B1": "Before Image 1", "B2": "Before Image 2", "A": "After Image"}
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(10)

        for col, key in enumerate(["B1", "B2", "A"]):
            card = QWidget()
            card.setFixedHeight(285)
            card.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(8, 8, 8, 8)
            card_layout.setSpacing(6)

            preview_label = QLabel(self.image_key_labels[key])
            preview_label.setAlignment(Qt.AlignCenter)
            preview_label.setFixedHeight(300)
            preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            preview_label.setStyleSheet("border: 1px solid #888; background: #262626;")
            card_layout.addWidget(preview_label)

            button = QPushButton(f"Load {self.image_key_labels[key]}")
            button.clicked.connect(lambda checked, value=key: self._open_image_for_key(value))
            button.setFixedHeight(30)
            card_layout.addWidget(button)

            path_label = QLabel("No file selected")
            path_label.setWordWrap(True)
            path_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            path_label.setAlignment(Qt.AlignTop)
            card_layout.addWidget(path_label)

            details_label = QLabel("No image details")
            details_label.setWordWrap(True)
            details_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            details_label.setAlignment(Qt.AlignLeft)
            details_label.setStyleSheet("font-size: 10px; color: #d0d0d0;")
            card_layout.addWidget(details_label)

            self.load_preview_labels[key] = preview_label
            self.path_labels[key] = path_label
            self.image_details_labels[key] = details_label
            grid.addWidget(card, 0, col)

        layout.addLayout(grid)
        return page

    def _get_intersection_masks_for_alignment(self):
        """Return the intersection paper and ink masks across all loaded images when alignment mode is enabled."""
        mask_papers = []
        mask_inks = []

        for key in ["B1", "B2", "A"]:
            image_data = self.images[key]
            if image_data.pixels.get("rgb") is None:
                return None, None

            paper_mask = self._as_bool_mask(image_data.masks.get("paper"))
            ink_mask = self._as_bool_mask(image_data.masks.get("ink"))
            array = np.asarray(image_data.get_pixels("rgb"))
            h, w = array.shape[:2]

            if paper_mask is None:
                paper_mask = np.ones((h, w), dtype=bool)
            if ink_mask is None:
                ink_mask = np.ones((h, w), dtype=bool)

            mask_papers.append(np.asarray(paper_mask, dtype=bool))
            mask_inks.append(np.asarray(ink_mask, dtype=bool))

        if not mask_papers:
            return None, None

        return np.logical_and.reduce(mask_papers), np.logical_and.reduce(mask_inks)

    def _display_alignment_image(self):
        """Show one of the three images in the alignment check area."""
        key = self.current_alignment_view
        image_data = self.images[key]
        image = image_data.get_pixels("rgb") if image_data.pixels.get("rgb") is not None else None
        if image is None:
            self.alignment_preview.setText(f"{key} not loaded")
            self.alignment_preview.setPixmap(QPixmap())
            return

        mode = self.alignment_display_mode
        if mode == "original":
            self._preview_from_array(self.alignment_preview, image)
            return

        if not self.use_mask_checkbox.isChecked():
            self._preview_from_array(self.alignment_preview, image)
            return

        if self.alignment_checkbox.isChecked():
            paper_mask, ink_mask = self._get_intersection_masks_for_alignment()
            self.analysis_options.int_paper_mask = paper_mask
            self.analysis_options.int_ink_mask = ink_mask
            if paper_mask is None or ink_mask is None:
                self._preview_from_array(self.alignment_preview, image)
                return
        else:
            self.analysis_options.int_paper_mask = None
            self.analysis_options.int_ink_mask = None
            self.analysis_options.intersection_mask = False
            paper_mask = image_data.masks.get("paper")
            ink_mask = image_data.masks.get("ink")
            if paper_mask is None or ink_mask is None:
                self._preview_from_array(self.alignment_preview, image)
                return

        if mode == "mask":
            self._display_mask_preview(
                self.alignment_preview,
                paper_mask=paper_mask,
                ink_mask=ink_mask,
            )
            return

        array = np.asarray(image, dtype=np.float32)
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=-1)
        if array.shape[-1] == 1:
            array = np.repeat(array, 3, axis=-1)
        array = np.clip(array, 0.0, 1.0)

        paper_bool = self._as_bool_mask(paper_mask)
        ink_bool = self._as_bool_mask(ink_mask)
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
        self.alignment_preview.setPixmap(pixmap, preserve_zoom=True)

    def _cycle_alignment_image(self):
        order = ["B1", "B2", "A"]
        current_index = order.index(self.current_alignment_view)
        self.current_alignment_view = order[(current_index + 1) % len(order)]
        self._update_alignment_buttons()
        self._display_alignment_image()

    def _images_have_identical_resolution(self):
        sizes = []
        for key in ["B1", "B2", "A"]:
            image_data = self.images[key]
            image = image_data.pixels.get("rgb")
            if image is None:
                return False
            array = np.asarray(image)
            if array.ndim == 0:
                return False
            sizes.append(tuple(array.shape[:2]))
        return len(set(sizes)) == 1

    def _update_alignment_mode_availability(self):
        if not hasattr(self, "alignment_checkbox"):
            return

        available = self._images_have_identical_resolution()
        self.alignment_checkbox.setEnabled(available)
        if not available:
            self.alignment_checkbox.setChecked(False)

    def _update_alignment_buttons(self):
        for key, button in self.alignment_buttons.items():
            button.setChecked(key == self.current_alignment_view)

    def _reset_alignment_zoom(self):
        if hasattr(self, "alignment_preview") and self.alignment_preview._base_pixmap is not None and not self.alignment_preview._base_pixmap.isNull():
            self.alignment_preview._auto_fit_zoom = True
            self.alignment_preview._apply_fit_zoom()
            self.alignment_preview._refresh_zoomed_pixmap()

    def _create_check_alignment_tab(self):
        """Tab 3: visually inspect the alignment by switching the view between B1/B2/A."""
        page = QWidget()
        page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop)

        self.alignment_checkbox = QCheckBox("Aligned image mode")
        self.alignment_checkbox.setChecked(False)
        self.alignment_checkbox.setEnabled(False)
        self.analysis_options.images_aligned = self.alignment_checkbox.isChecked()
        self.alignment_checkbox.toggled.connect(
            lambda checked: (
                self.analysis_options.sync_from_ui(images_aligned=checked),
                self._refresh_analysis_options_state(),
                self._display_alignment_image(),
            )
        )
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

        self.overlay_button = QPushButton(f"Mask overlay colour: {self.mask_overlay_colour.name()}")
        self.overlay_button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.overlay_button.clicked.connect(self._choose_mask_overlay_colour)
        self._update_overlay_button_style()
        self._update_alignment_mode_availability()
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

        self.alignment_preview = ClickableLabel("B1")
        self.alignment_preview.setAlignment(Qt.AlignCenter)
        self.alignment_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.alignment_preview.setMinimumSize(200, 120)
        self.alignment_preview.setStyleSheet("border: 1px solid #888; background: #262626;")
        self.alignment_preview.clicked.connect(self._cycle_alignment_image)

        self.reset_alignment_zoom_button = QPushButton("Reset zoom")
        self.reset_alignment_zoom_button.clicked.connect(self._reset_alignment_zoom)

        selector_row = QHBoxLayout()
        selector_row.setAlignment(Qt.AlignLeft)
        selector_row.addLayout(button_row)
        selector_row.addStretch(1)
        selector_row.addWidget(self.reset_alignment_zoom_button)
        layout.addLayout(selector_row)
        layout.addWidget(self.alignment_preview)

        self._update_alignment_buttons()
        self._refresh_alignment_preview()
        self._reset_alignment_zoom()
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
        image_data = self.images[key]
        colour = QColorDialog.getColor(image_data.reference_colours[region], self, f"Select {region} colour for {key}")
        if not colour.isValid():
            return

        image_data.reference_colours[region] = colour
        self.mask_reference_colours[key] = image_data.reference_colours
        self._update_reference_colour_buttons()

    def _change_all_reference_colours(self, region: str):
        representative = self.images["B1"].reference_colours[region]
        colour = QColorDialog.getColor(representative, self, f"Select all {region} colours")
        if not colour.isValid():
            return

        for key in ["B1", "B2", "A"]:
            self.images[key].reference_colours[region] = colour
            self.mask_reference_colours[key] = self.images[key].reference_colours

        self._update_reference_colour_buttons()

    def _update_reference_colour_buttons(self):
        for key in ["B1", "B2", "A"]:
            paper_colour = self.images[key].reference_colours["paper"]
            ink_colour = self.images[key].reference_colours["ink"]
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
        if not self._ensure_image_loaded_for_key(key):
            QMessageBox.warning(self, "Missing image", f"Load {key} before generating a mask.")
            return

        image_data = self.images[key]
        paper_mask, ink_mask, border_mask = image_data.generate_masks(
            self.mask_mode,
            threshold=self.mask_threshold_slider.value(),
            reference_weight=self.reference_weight_slider.value() / 100.0,
            border_pixels=self.mask_border_slider.value(),
        )

        if paper_mask is None or ink_mask is None:
            return

        if image_data.masks.get("ink") is not None:
            self._display_mask_preview(
                self.mask_preview_labels[key],
                paper_mask=image_data.masks.get("paper"),
                ink_mask=image_data.masks.get("ink"),
                border_mask=image_data.mask_border,
            )

        self._display_alignment_image()

    def _generate_mask_for_all(self):
        """Generate masks for all images currently loaded into the app."""
        for key in ["B1", "B2", "A"]:
            if self.images[key].pixels.get("rgb") is not None:
                self._generate_mask_for_key(key)

    def _remove_mask_for_key(self, key: str):
        """Clear the saved mask for a single image, leaving the original image visible."""
        image_data = self.images[key]
        image_data.clear_masks()

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
        settings_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(6)
        settings_layout.setContentsMargins(8, 8, 8, 8)

        mode_group = QGroupBox("Mask generation")
        mode_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        mode_group.setMaximumWidth(320)
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
        self.mask_threshold_slider.setFixedHeight(18)
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
        self.reference_weight_slider.setFixedHeight(18)
        self.reference_weight_label = QLabel("Reference weight: 0.50")
        self.reference_weight_slider.valueChanged.connect(
            lambda value: self.reference_weight_label.setText(f"Reference weight: {value / 100.0:.2f}")
        )
        settings_layout.addWidget(self.reference_weight_label)
        settings_layout.addWidget(self.reference_weight_slider)

        self.mask_border_slider = QSlider(Qt.Horizontal)
        self.mask_border_slider.setRange(0, 20)
        self.mask_border_slider.setValue(2)
        self.mask_border_slider.setFixedHeight(18)
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
        grid = QGridLayout()
        grid.setAlignment(Qt.AlignTop)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        for col, key in enumerate(["B1", "B2", "A"]):
            card = QWidget()
            card_layout = QVBoxLayout(card)

            original_preview = ZoomableLabel(f"Original {key}")
            original_preview.setAlignment(Qt.AlignCenter)
            original_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            original_preview.setStyleSheet("border: 1px solid #888; background: #262626;")
            card_layout.addWidget(original_preview)

            mask_preview = QLabel(f"Mask {key}")
            mask_preview.setAlignment(Qt.AlignCenter)
            mask_preview.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            mask_preview.setStyleSheet("border: 1px solid #888; background: #262626;")
            card_layout.addWidget(mask_preview)

            colour_row = QHBoxLayout()
            paper_button = QPushButton("Paper colour")
            paper_button.setMinimumWidth(90)
            paper_button.clicked.connect(lambda checked, value=key: self._set_mask_reference_colour(value, "paper"))
            ink_button = QPushButton("Ink colour")
            ink_button.setMinimumWidth(90)
            ink_button.clicked.connect(lambda checked, value=key: self._set_mask_reference_colour(value, "ink"))
            colour_row.addWidget(paper_button)
            colour_row.addWidget(ink_button)
            card_layout.addLayout(colour_row)

            button = QPushButton(f"Generate mask for {key}")
            button.setMinimumWidth(120)
            button.clicked.connect(lambda checked, value=key: self._generate_mask_for_key(value))
            card_layout.addWidget(button)

            remove_button = QPushButton(f"Remove mask for {key}")
            remove_button.setMinimumWidth(120)
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
        self.change_all_paper_button.setMinimumWidth(180)
        self.change_all_paper_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.change_all_paper_button.clicked.connect(lambda: self._change_all_reference_colours("paper"))
        self.change_all_ink_button = QPushButton("Set all ink colours")
        self.change_all_ink_button.setMinimumWidth(180)
        self.change_all_ink_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.change_all_ink_button.clicked.connect(lambda: self._change_all_reference_colours("ink"))

        self.generate_all_masks_button = QPushButton("Generate masks for all")
        self.generate_all_masks_button.setMinimumWidth(180)
        self.generate_all_masks_button.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.generate_all_masks_button.clicked.connect(self._generate_mask_for_all)

        bulk_actions_row = QHBoxLayout()
        bulk_actions_row.setContentsMargins(0, 0, 0, 0)
        bulk_actions_row.setAlignment(Qt.AlignLeft)
        bulk_actions_row.addWidget(self.change_all_paper_button)
        bulk_actions_row.addWidget(self.change_all_ink_button)
        bulk_actions_row.addWidget(self.generate_all_masks_button)
        bulk_actions_row.addStretch(1)
        settings_layout.addLayout(bulk_actions_row)

        layout.addLayout(grid)
        self._update_reference_colour_buttons()
        self._update_mask_generation_button_text()
        self._set_mask_controls_enabled(self.use_mask_checkbox.isChecked())
        self._set_mask_mode("luminance")
        return page

    def _refresh_analysis_options_state(self):
        """Keep the analysis option controls and their enabled states in sync with the stored settings."""
        if not hasattr(self, "analysis_options"):
            return

        options = self.analysis_options
        widgets = [
            ("rgb_comparison_checkbox", options.rgb_comparison),
            ("lab_comparison_checkbox", options.lab_comparison),
            ("use_masks_checkbox", options.use_masks),
            ("intersection_mask_checkbox", options.intersection_mask),
            ("mean_comparison_checkbox", options.mean_comparison),
            ("per_pixel_comparison_checkbox", options.per_pixel_comparison),
        ]
        for attr, value in widgets:
            if hasattr(self, attr):
                widget = getattr(self, attr)
                widget.blockSignals(True)
                widget.setChecked(value)
                widget.blockSignals(False)

        if hasattr(self, "heatmap_grid_slider"):
            slider = self.heatmap_grid_slider
            slider.blockSignals(True)
            slider.setValue(options.heatmap_grid_size)
            slider.blockSignals(False)

        intersection_enabled = options.use_masks and options.images_aligned
        per_pixel_enabled = not (options.use_masks and not options.intersection_mask)
        heatmap_enabled = options.per_pixel_comparison and per_pixel_enabled

        if hasattr(self, "intersection_mask_checkbox"):
            if not intersection_enabled:
                self.analysis_options.intersection_mask = False
                self.intersection_mask_checkbox.blockSignals(True)
                self.intersection_mask_checkbox.setChecked(False)
                self.intersection_mask_checkbox.blockSignals(False)
            self.intersection_mask_checkbox.setEnabled(intersection_enabled)

        if hasattr(self, "per_pixel_comparison_checkbox"):
            if not per_pixel_enabled:
                self.analysis_options.per_pixel_comparison = False
                self.per_pixel_comparison_checkbox.blockSignals(True)
                self.per_pixel_comparison_checkbox.setChecked(False)
                self.per_pixel_comparison_checkbox.blockSignals(False)
            self.per_pixel_comparison_checkbox.setEnabled(per_pixel_enabled)

        if hasattr(self, "heatmap_grid_slider"):
            if not heatmap_enabled:
                self.heatmap_grid_slider.setEnabled(False)
            else:
                self.heatmap_grid_slider.setEnabled(True)

    def _create_analysis_options_tab(self):
        """Tab 4: collect comparison and masking settings for later analysis."""
        page = QWidget()
        page.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        layout = QVBoxLayout(page)
        layout.setAlignment(Qt.AlignTop)

        options_group = QGroupBox("Comparison settings")
        options_group.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        options_layout = QVBoxLayout(options_group)

        self.rgb_comparison_checkbox = QCheckBox("RGB comparison")
        self.rgb_comparison_checkbox.setChecked(self.analysis_options.rgb_comparison)
        self.rgb_comparison_checkbox.toggled.connect(
            lambda checked: self.analysis_options.sync_from_ui(rgb_comparison=checked)
        )
        options_layout.addWidget(self.rgb_comparison_checkbox)

        self.lab_comparison_checkbox = QCheckBox("LAB comparison")
        self.lab_comparison_checkbox.setChecked(self.analysis_options.lab_comparison)
        self.lab_comparison_checkbox.toggled.connect(
            lambda checked: self.analysis_options.sync_from_ui(lab_comparison=checked)
        )
        options_layout.addWidget(self.lab_comparison_checkbox)

        self.use_masks_checkbox = QCheckBox("Use masks")
        self.use_masks_checkbox.setChecked(self.analysis_options.use_masks)
        self.use_masks_checkbox.toggled.connect(
            lambda checked: (
                self.analysis_options.sync_from_ui(use_masks=checked),
                self._refresh_analysis_options_state(),
            )
        )
        options_layout.addWidget(self.use_masks_checkbox)

        self.intersection_mask_checkbox = QCheckBox("Intersection mask")
        self.intersection_mask_checkbox.setChecked(self.analysis_options.intersection_mask)
        self.intersection_mask_checkbox.toggled.connect(
            lambda checked: (
                self.analysis_options.sync_from_ui(intersection_mask=checked),
                self._refresh_analysis_options_state(),
            )
        )
        options_layout.addWidget(self.intersection_mask_checkbox)

        self.mean_comparison_checkbox = QCheckBox("Mean comparison")
        self.mean_comparison_checkbox.setChecked(self.analysis_options.mean_comparison)
        self.mean_comparison_checkbox.toggled.connect(
            lambda checked: self.analysis_options.sync_from_ui(mean_comparison=checked)
        )
        options_layout.addWidget(self.mean_comparison_checkbox)

        self.per_pixel_comparison_checkbox = QCheckBox("Per-pixel comparison")
        self.per_pixel_comparison_checkbox.setChecked(self.analysis_options.per_pixel_comparison)
        self.per_pixel_comparison_checkbox.toggled.connect(
            lambda checked: (
                self.analysis_options.sync_from_ui(per_pixel_comparison=checked),
                self._refresh_analysis_options_state(),
            )
        )
        options_layout.addWidget(self.per_pixel_comparison_checkbox)

        self.heatmap_grid_slider = QSlider(Qt.Horizontal)
        self.heatmap_grid_slider.setRange(1, 100)
        self.heatmap_grid_slider.setValue(self.analysis_options.heatmap_grid_size)
        self.heatmap_grid_slider.valueChanged.connect(
            lambda value: self.analysis_options.sync_from_ui(heatmap_grid_size=value)
        )

        heatmap_label = QLabel(f"Heatmap grid size: {self.analysis_options.heatmap_grid_size}px")
        self.heatmap_grid_slider.valueChanged.connect(
            lambda value: heatmap_label.setText(f"Heatmap grid size: {value}px")
        )
        options_layout.addWidget(heatmap_label)
        options_layout.addWidget(self.heatmap_grid_slider)

        layout.addWidget(options_group)
        self._refresh_analysis_options_state()
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

    def dict_to_string(self,d, indent=4, _level=0):
        """Convert a nested dict to an indented string.

        Keys with dict values become headers ("key:") with their children
        indented beneath them. Any other value is written inline ("key: value").
        """
        pad = " " * (indent * _level)
        lines = []
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{pad}{key}:")
                if value:  # skip empty dicts to avoid blank lines
                    lines.append(self.dict_to_string(value, indent, _level + 1))
            else:
                lines.append(f"{pad}{key}: {value}")
        return "\n".join(lines)

    def analyze_images(self):
        """Run the comparison for the three loaded images and report the result."""
        
        if any(self.images[key].pixels.get("rgb") is None for key in ["B1", "B2", "A"]):
            QMessageBox.warning(self, "Missing input", "Load B1, B2, and A before running the analysis.")
            return
        try:
            
            results = compare_images(self.images["B1"], self.images["B2"], self.images["A"], self.analysis_options)
            
            summary_lines = self.dict_to_string(results,6)

            self.report.setPlainText(summary_lines)
            
            self.status_label.setText("Analysis complete. The report compares B1 against both B2 and A.")
            
        except Exception as exc:
            QMessageBox.critical(self, "Analysis failed", f"Unable to analyze the images:\n{exc}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
