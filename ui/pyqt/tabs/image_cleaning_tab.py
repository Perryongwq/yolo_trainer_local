"""Image Cleaning tab – resize, grayscale, binarize, and batch-export images."""

import os
import math
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QComboBox, QLineEdit, QCheckBox, QSlider,
    QFileDialog, QProgressBar, QScrollArea, QSplitter, QFrame,
    QSizePolicy, QMessageBox,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIntValidator

from ui.pyqt.common.ui_utils import main_window_parent
from utils.image_rendering import cv_to_pixmap, fit_pixmap

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageCleaningTab:
    """Self-contained tab for basic image pre-processing."""

    def __init__(self, ui, logger):
        self.ui = ui
        self.logger = logger

        # State
        self._image_files: list[str] = []
        self._image_index: int = -1
        self._original_image: np.ndarray | None = None
        self._processed_image: np.ndarray | None = None
        self._pre_binarize_image: np.ndarray | None = None
        self._show_original: bool = False
        self._output_folder: str = ""

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        """Programmatically build the tab and add it to the main tab widget."""
        self._tab_widget = QWidget()
        self._tab_widget.setObjectName("tab_image_cleaning")

        root_layout = QHBoxLayout(self._tab_widget)
        splitter = QSplitter(Qt.Horizontal, self._tab_widget)
        root_layout.addWidget(splitter)

        # --- Left: scrollable controls ---
        screen = QApplication.primaryScreen().availableGeometry()
        ctrl_width = max(380, min(int(screen.width() * 0.30), 600))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(ctrl_width)
        scroll.setMaximumWidth(ctrl_width)
        controls = QWidget()
        controls.setMinimumWidth(ctrl_width - 20)  # account for scrollbar
        self._controls_layout = QVBoxLayout(controls)
        self._controls_layout.setAlignment(Qt.AlignTop)

        self._build_source_group()
        self._btn_reset = QPushButton("Reset Image")
        self._controls_layout.addWidget(self._btn_reset)
        self._build_info_group()
        self._build_resize_group()
        self._build_grayscale_group()
        self._build_binarize_group()
        self._build_output_group()

        self._controls_layout.addStretch()
        scroll.setWidget(controls)
        splitter.addWidget(scroll)

        # --- Right: image preview ---
        preview_frame = QFrame()
        preview_layout = QVBoxLayout(preview_frame)
        preview_layout.setAlignment(Qt.AlignCenter)
        self._preview_label = QLabel("No image loaded")
        self._preview_label.setAlignment(Qt.AlignCenter)
        self._preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        preview_layout.addWidget(self._preview_label)
        splitter.addWidget(preview_frame)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Register as new tab — insert before Model Training (index 0)
        self.ui.tabWidget.insertTab(0, self._tab_widget, "Image Cleaning")
        self.ui.tabWidget.setCurrentIndex(0)

    # --- Source group ---

    def _build_source_group(self):
        grp = QGroupBox("Image Source")
        lay = QVBoxLayout(grp)

        row1 = QHBoxLayout()
        self._btn_browse_folder = QPushButton("Browse Folder")
        self._btn_open_single = QPushButton("Open Single Image")
        row1.addWidget(self._btn_browse_folder)
        row1.addWidget(self._btn_open_single)
        lay.addLayout(row1)

        row2 = QHBoxLayout()
        self._btn_prev = QPushButton("◀ Previous")
        self._lbl_nav = QLabel("0 / 0")
        self._lbl_nav.setAlignment(Qt.AlignCenter)
        self._btn_next = QPushButton("Next ▶")
        row2.addWidget(self._btn_prev)
        row2.addWidget(self._lbl_nav)
        row2.addWidget(self._btn_next)
        lay.addLayout(row2)

        self._controls_layout.addWidget(grp)

    # --- Info group ---

    def _build_info_group(self):
        grp = QGroupBox("Image Info")
        lay = QVBoxLayout(grp)
        self._lbl_dimensions = QLabel("Dimensions: —")
        self._lbl_aspect = QLabel("Aspect Ratio: —")
        self._lbl_intensity = QLabel("Mean Intensity: —")
        lay.addWidget(self._lbl_dimensions)
        lay.addWidget(self._lbl_aspect)
        lay.addWidget(self._lbl_intensity)
        self._controls_layout.addWidget(grp)

    # --- Resize group ---

    def _build_resize_group(self):
        grp = QGroupBox("Resize")
        lay = QVBoxLayout(grp)

        dim_row = QHBoxLayout()
        dim_row.addWidget(QLabel("W:"))
        self._txt_width = QLineEdit()
        self._txt_width.setValidator(QIntValidator(1, 99999))
        dim_row.addWidget(self._txt_width)
        dim_row.addWidget(QLabel("H:"))
        self._txt_height = QLineEdit()
        self._txt_height.setValidator(QIntValidator(1, 99999))
        dim_row.addWidget(self._txt_height)
        lay.addLayout(dim_row)

        self._chk_lock_ratio = QCheckBox("Lock aspect ratio")
        self._chk_lock_ratio.setChecked(True)
        lay.addWidget(self._chk_lock_ratio)

        preset_row = QHBoxLayout()
        for size in (128, 224, 256, 512):
            btn = QPushButton(f"{size}×{size}")
            btn.clicked.connect(lambda checked, s=size: self._apply_preset(s))
            preset_row.addWidget(btn)
        lay.addLayout(preset_row)

        algo_row = QHBoxLayout()
        algo_row.addWidget(QLabel("Algorithm:"))
        self._cmb_interp = QComboBox()
        self._cmb_interp.addItems(["Bilinear", "Nearest Neighbor"])
        algo_row.addWidget(self._cmb_interp)
        lay.addLayout(algo_row)

        self._btn_resize = QPushButton("Apply Resize")
        lay.addWidget(self._btn_resize)
        self._controls_layout.addWidget(grp)

    # --- Grayscale group ---

    def _build_grayscale_group(self):
        grp = QGroupBox("Grayscale Conversion")
        lay = QVBoxLayout(grp)
        lay.addWidget(QLabel("Formula: 0.299R + 0.587G + 0.114B"))
        btn_row = QHBoxLayout()
        self._btn_grayscale = QPushButton("Convert to Grayscale")
        self._btn_toggle_gray = QPushButton("Toggle Before / After")
        btn_row.addWidget(self._btn_grayscale)
        btn_row.addWidget(self._btn_toggle_gray)
        lay.addLayout(btn_row)
        self._controls_layout.addWidget(grp)

    # --- Binarization group ---

    def _build_binarize_group(self):
        grp = QGroupBox("Binarization")
        lay = QVBoxLayout(grp)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(QLabel("Threshold:"))
        self._slider_thresh = QSlider(Qt.Horizontal)
        self._slider_thresh.setRange(0, 255)
        self._slider_thresh.setValue(128)
        thresh_row.addWidget(self._slider_thresh)
        self._lbl_thresh_val = QLabel("128")
        self._lbl_thresh_val.setMinimumWidth(30)
        thresh_row.addWidget(self._lbl_thresh_val)
        lay.addLayout(thresh_row)

        btn_row = QHBoxLayout()
        self._btn_binarize = QPushButton("Apply Binarization")
        self._btn_toggle_bin = QPushButton("Toggle Before / After")
        btn_row.addWidget(self._btn_binarize)
        btn_row.addWidget(self._btn_toggle_bin)
        lay.addLayout(btn_row)
        self._controls_layout.addWidget(grp)

    # --- Output group ---

    def _build_output_group(self):
        grp = QGroupBox("Output")
        lay = QVBoxLayout(grp)

        folder_row = QHBoxLayout()
        self._txt_output_folder = QLineEdit()
        self._txt_output_folder.setReadOnly(True)
        self._txt_output_folder.setPlaceholderText("Select output folder…")
        self._btn_browse_output = QPushButton("Browse")
        folder_row.addWidget(self._txt_output_folder)
        folder_row.addWidget(self._btn_browse_output)
        lay.addLayout(folder_row)

        save_row = QHBoxLayout()
        self._btn_save_current = QPushButton("Save Current Image")
        self._btn_save_all = QPushButton("Save All Processed")
        save_row.addWidget(self._btn_save_current)
        save_row.addWidget(self._btn_save_all)
        lay.addLayout(save_row)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)

        self._controls_layout.addWidget(grp)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self._btn_browse_folder.clicked.connect(self._on_browse_folder)
        self._btn_open_single.clicked.connect(self._on_open_single)
        self._btn_prev.clicked.connect(self._on_prev)
        self._btn_next.clicked.connect(self._on_next)
        self._btn_reset.clicked.connect(self._on_reset)

        self._txt_width.editingFinished.connect(self._on_width_changed)
        self._txt_height.editingFinished.connect(self._on_height_changed)
        self._btn_resize.clicked.connect(self._on_resize)

        self._btn_grayscale.clicked.connect(self._on_grayscale)
        self._btn_toggle_gray.clicked.connect(self._toggle_before_after)

        self._slider_thresh.valueChanged.connect(
            lambda v: self._lbl_thresh_val.setText(str(v))
        )
        self._slider_thresh.valueChanged.connect(self._on_thresh_changed)
        self._btn_binarize.clicked.connect(self._on_binarize)
        self._btn_toggle_bin.clicked.connect(self._toggle_before_after)

        self._btn_browse_output.clicked.connect(self._on_browse_output)
        self._btn_save_current.clicked.connect(self._on_save_current)
        self._btn_save_all.clicked.connect(self._on_save_all)

    # ------------------------------------------------------------------
    # Image source handlers
    # ------------------------------------------------------------------

    def _on_browse_folder(self):
        parent = main_window_parent(self.logger)
        folder = QFileDialog.getExistingDirectory(parent, "Select Image Folder")
        if not folder:
            return
        files = sorted(
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in _IMAGE_EXTS
        )
        if not files:
            self.logger.warning("No image files found in selected folder.")
            return
        self._image_files = files
        self._image_index = 0
        self._load_current_image()
        self.logger.info(f"Loaded folder with {len(files)} image(s).")

    def _on_open_single(self):
        parent = main_window_parent(self.logger)
        path, _ = QFileDialog.getOpenFileName(
            parent, "Open Image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        self._image_files = [path]
        self._image_index = 0
        self._load_current_image()

    def _on_reset(self):
        if self._original_image is None:
            return
        self._processed_image = self._original_image.copy()
        self._pre_binarize_image = None
        self._show_original = False
        self._update_info()
        self._refresh_preview()
        self.logger.info("Image reset to original.")

    def _on_prev(self):
        if self._image_files and self._image_index > 0:
            self._image_index -= 1
            self._load_current_image()

    def _on_next(self):
        if self._image_files and self._image_index < len(self._image_files) - 1:
            self._image_index += 1
            self._load_current_image()

    def _load_current_image(self):
        path = self._image_files[self._image_index]
        img = cv2.imread(path)
        if img is None:
            self.logger.error(f"Failed to read image: {path}")
            return
        self._original_image = img
        self._processed_image = img.copy()
        self._pre_binarize_image = None
        self._show_original = False
        self._update_info()
        self._update_nav_label()
        self._refresh_preview()

    # ------------------------------------------------------------------
    # Info display
    # ------------------------------------------------------------------

    def _update_info(self):
        img = self._processed_image
        if img is None:
            return
        h, w = img.shape[:2]
        self._lbl_dimensions.setText(f"Dimensions: {w} × {h} px")

        g = math.gcd(w, h)
        self._lbl_aspect.setText(f"Aspect Ratio: {w // g}:{h // g}")

        mean_val = np.mean(img) / 255.0
        self._lbl_intensity.setText(f"Mean Intensity: {mean_val:.3f}")

        self._txt_width.setText(str(w))
        self._txt_height.setText(str(h))

    def _update_nav_label(self):
        total = len(self._image_files)
        cur = self._image_index + 1 if total else 0
        self._lbl_nav.setText(f"{cur} / {total}")

    # ------------------------------------------------------------------
    # Resize handlers
    # ------------------------------------------------------------------

    def _apply_preset(self, size: int):
        self._chk_lock_ratio.setChecked(False)
        self._txt_width.setText(str(size))
        self._txt_height.setText(str(size))

    def _on_width_changed(self):
        if not self._chk_lock_ratio.isChecked() or self._original_image is None:
            return
        oh, ow = self._original_image.shape[:2]
        try:
            new_w = int(self._txt_width.text())
        except ValueError:
            return
        new_h = max(1, round(new_w * oh / ow))
        self._txt_height.blockSignals(True)
        self._txt_height.setText(str(new_h))
        self._txt_height.blockSignals(False)

    def _on_height_changed(self):
        if not self._chk_lock_ratio.isChecked() or self._original_image is None:
            return
        oh, ow = self._original_image.shape[:2]
        try:
            new_h = int(self._txt_height.text())
        except ValueError:
            return
        new_w = max(1, round(new_h * ow / oh))
        self._txt_width.blockSignals(True)
        self._txt_width.setText(str(new_w))
        self._txt_width.blockSignals(False)

    def _on_resize(self):
        if self._processed_image is None:
            return
        try:
            new_w = int(self._txt_width.text())
            new_h = int(self._txt_height.text())
        except ValueError:
            self.logger.warning("Invalid width/height values.")
            return
        interp = (
            cv2.INTER_LINEAR
            if self._cmb_interp.currentIndex() == 0
            else cv2.INTER_NEAREST
        )
        self._processed_image = cv2.resize(self._processed_image, (new_w, new_h), interpolation=interp)
        self._update_info()
        self._refresh_preview()
        self.logger.info(f"Resized to {new_w}×{new_h}.")

    # ------------------------------------------------------------------
    # Grayscale handler
    # ------------------------------------------------------------------

    def _on_grayscale(self):
        if self._processed_image is None:
            return
        img = self._processed_image
        if len(img.shape) == 2:
            self.logger.info("Image is already grayscale.")
            return
        # Luminosity formula: BGR -> 0.114B + 0.587G + 0.299R
        gray = (0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]).astype(np.uint8)
        self._processed_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        self._update_info()
        self._refresh_preview()
        self.logger.info("Converted to grayscale (luminosity).")

    # ------------------------------------------------------------------
    # Binarize handler
    # ------------------------------------------------------------------

    def _on_binarize(self):
        if self._processed_image is None:
            return
        # Snapshot before first binarize so slider can re-apply non-destructively
        if self._pre_binarize_image is None:
            self._pre_binarize_image = self._processed_image.copy()
        self._apply_binarize()

    def _on_thresh_changed(self, _value):
        """Live-update binarization when the slider moves (only if already binarizing)."""
        if self._pre_binarize_image is None:
            return
        self._apply_binarize()

    def _apply_binarize(self):
        img = self._pre_binarize_image
        thresh = self._slider_thresh.value()
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        _, binary = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
        self._processed_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        self._update_info()
        self._refresh_preview()

    # ------------------------------------------------------------------
    # Toggle before / after
    # ------------------------------------------------------------------

    def _toggle_before_after(self):
        if self._original_image is None:
            return
        self._show_original = not self._show_original
        self._refresh_preview()

    # ------------------------------------------------------------------
    # Preview
    # ------------------------------------------------------------------

    def _refresh_preview(self):
        img = self._original_image if self._show_original else self._processed_image
        if img is None:
            return
        # Ensure 3-channel for cv_to_pixmap
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pixmap = cv_to_pixmap(rgb)
        max_w = self._preview_label.width()
        max_h = self._preview_label.height()
        if max_w > 0 and max_h > 0:
            pixmap = fit_pixmap(pixmap, max_w, max_h)
        self._preview_label.setPixmap(pixmap)

    # ------------------------------------------------------------------
    # Output handlers
    # ------------------------------------------------------------------

    def _on_browse_output(self):
        parent = main_window_parent(self.logger)
        folder = QFileDialog.getExistingDirectory(parent, "Select Output Folder")
        if folder:
            self._output_folder = folder
            self._txt_output_folder.setText(folder)

    def _on_save_current(self):
        if self._processed_image is None:
            self.logger.warning("No image to save.")
            return
        if not self._output_folder:
            self._on_browse_output()
            if not self._output_folder:
                return
        src_name = os.path.basename(self._image_files[self._image_index])
        out_path = os.path.join(self._output_folder, src_name)
        cv2.imwrite(out_path, self._processed_image)
        self.logger.info(f"Saved: {out_path}")

    def _on_save_all(self):
        if not self._image_files:
            self.logger.warning("No images loaded.")
            return
        if not self._output_folder:
            self._on_browse_output()
            if not self._output_folder:
                return

        # Infer pipeline from current processed vs original state
        pipeline = self._infer_pipeline()
        total = len(self._image_files)

        self._progress.setRange(0, total)
        self._progress.setValue(0)
        self._progress.setVisible(True)

        for i, path in enumerate(self._image_files):
            img = cv2.imread(path)
            if img is None:
                self.logger.warning(f"Skipping unreadable file: {path}")
                self._progress.setValue(i + 1)
                continue

            img = self._apply_pipeline(img, pipeline)
            out_path = os.path.join(self._output_folder, os.path.basename(path))
            cv2.imwrite(out_path, img)
            self._progress.setValue(i + 1)

        self._progress.setVisible(False)
        self.logger.info(f"Batch saved {total} image(s) to {self._output_folder}")

    # ------------------------------------------------------------------
    # Pipeline inference for batch save
    # ------------------------------------------------------------------

    def _infer_pipeline(self) -> dict:
        """Infer processing steps from the current image state."""
        pipe: dict = {}
        if self._original_image is None or self._processed_image is None:
            return pipe
        oh, ow = self._original_image.shape[:2]
        ph, pw = self._processed_image.shape[:2]
        if (pw, ph) != (ow, oh):
            pipe["resize"] = (pw, ph)
            pipe["interp"] = (
                cv2.INTER_LINEAR
                if self._cmb_interp.currentIndex() == 0
                else cv2.INTER_NEAREST
            )
        # Check if grayscale (all channels equal in BGR representation)
        proc = self._processed_image
        if len(proc.shape) == 3 and np.array_equal(proc[:, :, 0], proc[:, :, 1]):
            pipe["grayscale"] = True
            # Check binary (only 0 and 255)
            unique = np.unique(proc[:, :, 0])
            if len(unique) <= 2 and all(v in (0, 255) for v in unique):
                pipe["binarize"] = self._slider_thresh.value()
        return pipe

    def _apply_pipeline(self, img: np.ndarray, pipeline: dict) -> np.ndarray:
        if "resize" in pipeline:
            w, h = pipeline["resize"]
            img = cv2.resize(img, (w, h), interpolation=pipeline.get("interp", cv2.INTER_LINEAR))
        if "grayscale" in pipeline:
            if len(img.shape) == 3:
                gray = (0.114 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.299 * img[:, :, 2]).astype(np.uint8)
            else:
                gray = img
            if "binarize" in pipeline:
                _, img = cv2.threshold(gray, pipeline["binarize"], 255, cv2.THRESH_BINARY)
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            else:
                img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        return img
