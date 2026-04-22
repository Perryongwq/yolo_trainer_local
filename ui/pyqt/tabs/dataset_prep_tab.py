"""Dataset Preparation tab – split image+JSON pairs into train/val sets."""

import json
import os
import random
import shutil
from pathlib import Path

import yaml

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QSlider, QProgressBar, QScrollArea, QSplitter,
    QFrame, QSpinBox, QTextEdit, QMessageBox, QFileDialog, QRadioButton,
    QButtonGroup, QDialog, QDialogButtonBox, QFormLayout,
)

from ui.pyqt.common.ui_utils import main_window_parent

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------------------
# Background worker
# ---------------------------------------------------------------------------

class _SplitWorker(QThread):
    """Runs the train/val split in a background thread."""

    progress = pyqtSignal(int, int)   # (current, total)
    log      = pyqtSignal(str)        # log line
    finished = pyqtSignal(dict)       # summary dict

    def __init__(self, input_dir: str, output_dir: str,
                 train_ratio: float, seed: int):
        super().__init__()
        self._input_dir  = Path(input_dir)
        self._output_dir = Path(output_dir)
        self._train_ratio = train_ratio
        self._seed = seed

    def run(self):
        try:
            self._execute()
        except Exception as exc:
            self.log.emit(f"[ERROR] Unexpected error: {exc}")
            self.finished.emit({"error": str(exc)})

    def _execute(self):
        src = self._input_dir
        dst = self._output_dir

        # --- Scan pairs ---
        image_files = {
            f.stem: f for f in src.iterdir()
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        }
        json_files = {
            f.stem: f for f in src.iterdir()
            if f.is_file() and f.suffix.lower() == ".json"
        }

        paired_stems  = sorted(set(image_files) & set(json_files))
        skipped_stems = sorted(set(image_files) - set(json_files))

        self.log.emit(f"Images found     : {len(image_files)}")
        self.log.emit(f"JSON files found : {len(json_files)}")
        self.log.emit(f"Valid pairs      : {len(paired_stems)}")
        self.log.emit(f"Skipped (no JSON): {len(skipped_stems)}")
        if skipped_stems:
            for s in skipped_stems[:10]:
                self.log.emit(f"  ⚠ skipped: {image_files[s].name}")
            if len(skipped_stems) > 10:
                self.log.emit(f"  … and {len(skipped_stems) - 10} more")

        if not paired_stems:
            self.log.emit("[ERROR] No valid pairs found. Aborting.")
            self.finished.emit({"error": "No valid pairs"})
            return

        # --- Build label→class map across all JSONs ---
        label_to_class = self._build_label_map([json_files[s] for s in paired_stems])
        self.log.emit(f"Classes found    : {len(label_to_class)}")
        for lbl, cid in sorted(label_to_class.items(), key=lambda x: x[1]):
            self.log.emit(f"  {cid}: {lbl}")

        # --- Shuffle & split ---
        rng = random.Random(self._seed)
        ordered = list(paired_stems)
        rng.shuffle(ordered)
        split_idx   = max(1, int(len(ordered) * self._train_ratio))
        train_stems = ordered[:split_idx]
        val_stems   = ordered[split_idx:]

        self.log.emit(f"\nTrain set: {len(train_stems)} pairs")
        self.log.emit(f"Val set  : {len(val_stems)} pairs")

        # --- Create output dirs ---
        for sub in ("train/images", "train/labels", "val/images", "val/labels"):
            (dst / sub).mkdir(parents=True, exist_ok=True)

        # --- Copy & convert files ---
        total = len(ordered)
        done  = 0
        for split_name, stems in (("train", train_stems), ("val", val_stems)):
            for stem in stems:
                img_src  = image_files[stem]
                json_src = json_files[stem]

                shutil.copy2(img_src, dst / split_name / "images" / img_src.name)

                yolo_lines = self._convert_json(json_src, label_to_class)
                txt_dst = dst / split_name / "labels" / f"{stem}.txt"
                txt_dst.write_text(
                    "\n".join(yolo_lines) + ("\n" if yolo_lines else ""),
                    encoding="utf-8"
                )

                # Retain original JSON alongside the .txt
                shutil.copy2(json_src, dst / split_name / "labels" / json_src.name)

                done += 1
                self.progress.emit(done, total)

        # --- classes.txt at output root ---
        (dst / "classes.txt").write_text(
            "\n".join(
                lbl for lbl, _ in sorted(label_to_class.items(), key=lambda x: x[1])
            ) + "\n",
            encoding="utf-8"
        )
        self.log.emit(f"\nclasses.txt saved to {dst / 'classes.txt'}")
        self.log.emit("\n✓ Split complete.")

        class_names = [lbl for lbl, _ in sorted(label_to_class.items(), key=lambda x: x[1])]
        self.finished.emit({
            "train":       len(train_stems),
            "val":         len(val_stems),
            "skipped":     len(skipped_stems),
            "classes":     len(label_to_class),
            "class_names": class_names,
        })

    # ------------------------------------------------------------------

    @staticmethod
    def _build_label_map(json_paths: list) -> dict:
        labels: set = set()
        for p in json_paths:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                for shape in data.get("shapes", []):
                    lbl = shape.get("label", "").strip()
                    if lbl:
                        labels.add(lbl)
            except Exception:
                pass
        return {lbl: idx for idx, lbl in enumerate(sorted(labels))}

    @staticmethod
    def _convert_json(json_path: Path, label_to_class: dict) -> list:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            return []
        img_w = data.get("imageWidth")
        img_h = data.get("imageHeight")
        if not img_w or not img_h:
            return []
        lines = []
        for shape in data.get("shapes", []):
            label  = shape.get("label", "").strip()
            points = shape.get("points", [])
            if label not in label_to_class or not points:
                continue
            try:
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
            except (IndexError, TypeError):
                continue
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            cx = max(0.0, min(1.0, ((x1 + x2) / 2) / img_w))
            cy = max(0.0, min(1.0, ((y1 + y2) / 2) / img_h))
            bw = max(0.0, min(1.0, (x2 - x1) / img_w))
            bh = max(0.0, min(1.0, (y2 - y1) / img_h))
            lines.append(
                f"{label_to_class[label]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
            )
        return lines


# ---------------------------------------------------------------------------
# TXT-based worker (images already have YOLO .txt labels)
# ---------------------------------------------------------------------------

class _SplitWorkerTxt(QThread):
    """Split a folder of image + YOLO .txt label pairs into train/val."""

    progress = pyqtSignal(int, int)
    log      = pyqtSignal(str)
    finished = pyqtSignal(dict)

    def __init__(self, input_dir: str, output_dir: str,
                 train_ratio: float, seed: int):
        super().__init__()
        self._input_dir  = Path(input_dir)
        self._output_dir = Path(output_dir)
        self._train_ratio = train_ratio
        self._seed = seed

    def run(self):
        try:
            self._execute()
        except Exception as exc:
            self.log.emit(f"[ERROR] Unexpected error: {exc}")
            self.finished.emit({"error": str(exc)})

    def _execute(self):
        src = self._input_dir
        dst = self._output_dir

        image_files = {
            f.stem: f for f in src.iterdir()
            if f.is_file() and f.suffix.lower() in _IMAGE_EXTS
        }
        txt_files = {
            f.stem: f for f in src.iterdir()
            if f.is_file() and f.suffix.lower() == ".txt"
        }

        paired_stems  = sorted(set(image_files) & set(txt_files))
        skipped_stems = sorted(set(image_files) - set(txt_files))

        self.log.emit(f"Images found      : {len(image_files)}")
        self.log.emit(f"TXT labels found  : {len(txt_files)}")
        self.log.emit(f"Valid pairs       : {len(paired_stems)}")
        self.log.emit(f"Skipped (no TXT)  : {len(skipped_stems)}")
        if skipped_stems:
            for s in skipped_stems[:10]:
                self.log.emit(f"  \u26a0 skipped: {image_files[s].name}")
            if len(skipped_stems) > 10:
                self.log.emit(f"  \u2026 and {len(skipped_stems) - 10} more")

        if not paired_stems:
            self.log.emit("[ERROR] No valid image+TXT pairs found. Aborting.")
            self.finished.emit({"error": "No valid pairs"})
            return

        rng = random.Random(self._seed)
        ordered = list(paired_stems)
        rng.shuffle(ordered)
        split_idx   = max(1, int(len(ordered) * self._train_ratio))
        train_stems = ordered[:split_idx]
        val_stems   = ordered[split_idx:]

        self.log.emit(f"\nTrain set: {len(train_stems)} pairs")
        self.log.emit(f"Val set  : {len(val_stems)} pairs")

        for sub in ("train/images", "train/labels", "val/images", "val/labels"):
            (dst / sub).mkdir(parents=True, exist_ok=True)

        total = len(ordered)
        done  = 0
        for split_name, stems in (("train", train_stems), ("val", val_stems)):
            for stem in stems:
                img_src = image_files[stem]
                txt_src = txt_files[stem]
                shutil.copy2(img_src, dst / split_name / "images" / img_src.name)
                shutil.copy2(txt_src, dst / split_name / "labels" / txt_src.name)
                done += 1
                self.progress.emit(done, total)

        self.log.emit("\n\u2713 Split complete.")

        # Try to read class names from classes.txt in input folder
        class_names = []
        classes_txt = src / "classes.txt"
        if classes_txt.exists():
            class_names = [l.strip() for l in classes_txt.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.log.emit(f"classes.txt found — {len(class_names)} classes loaded.")
        else:
            self.log.emit("\u26a0 No classes.txt found in input folder — dataset.yaml names will be empty.")

        self.finished.emit({
            "train":       len(train_stems),
            "val":         len(val_stems),
            "skipped":     len(skipped_stems),
            "class_names": class_names,
        })


# ---------------------------------------------------------------------------
# Tab class
# ---------------------------------------------------------------------------

class DatasetPrepTab:
    """Self-contained Dataset Preparation tab (train/val split from LabelMe JSON)."""

    def __init__(self, ui, logger):
        self.ui = ui
        self.logger = logger
        self._worker: _SplitWorker | None = None

        self._build_ui()
        self._connect_signals()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        self._tab_widget = QWidget()
        self._tab_widget.setObjectName("tab_dataset_prep")

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
        controls.setMinimumWidth(ctrl_width - 20)
        self._ctrl_layout = QVBoxLayout(controls)
        self._ctrl_layout.setAlignment(Qt.AlignTop)

        self._build_input_group()
        self._build_format_group()
        self._build_split_group()
        self._build_output_group()
        self._build_run_group()

        self._ctrl_layout.addStretch()
        scroll.setWidget(controls)
        splitter.addWidget(scroll)

        # --- Right: results panel ---
        right = QFrame()
        right_layout = QVBoxLayout(right)

        stats_row = QHBoxLayout()
        self._card_paired  = self._stat_card("Pairs",   "—")
        self._card_skipped = self._stat_card("Skipped", "—")
        self._card_train   = self._stat_card("Train",   "—")
        self._card_val     = self._stat_card("Val",     "—")
        for card in (self._card_paired, self._card_skipped,
                     self._card_train, self._card_val):
            stats_row.addWidget(card)
        right_layout.addLayout(stats_row)

        log_grp = QGroupBox("Log")
        log_lay = QVBoxLayout(log_grp)
        self._txt_log = QTextEdit()
        self._txt_log.setReadOnly(True)
        self._txt_log.setPlaceholderText("Run output will appear here…")
        log_lay.addWidget(self._txt_log)
        right_layout.addWidget(log_grp)

        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        # Insert after Image Cleaning (index 0) → index 1
        self.ui.tabWidget.insertTab(1, self._tab_widget, "Dataset Preparation")

    @staticmethod
    def _stat_card(title: str, value: str) -> QGroupBox:
        grp = QGroupBox(title)
        lay = QVBoxLayout(grp)
        lbl = QLabel(value)
        lbl.setAlignment(Qt.AlignCenter)
        font = lbl.font()
        font.setPointSize(font.pointSize() + 4)
        font.setBold(True)
        lbl.setFont(font)
        lay.addWidget(lbl)
        grp._value_label = lbl  # type: ignore[attr-defined]
        return grp

    def _build_input_group(self):
        grp = QGroupBox("Input Folder")
        lay = QVBoxLayout(grp)
        row = QHBoxLayout()
        self._txt_input = QLineEdit()
        self._txt_input.setReadOnly(True)
        self._txt_input.setPlaceholderText("Select folder with images + labels\u2026")
        self._btn_browse_input = QPushButton("Browse")
        row.addWidget(self._txt_input)
        row.addWidget(self._btn_browse_input)
        lay.addLayout(row)
        self._btn_scan = QPushButton("Scan Folder")
        lay.addWidget(self._btn_scan)
        self._lbl_scan_result = QLabel("No folder selected.")
        self._lbl_scan_result.setWordWrap(True)
        lay.addWidget(self._lbl_scan_result)
        self._ctrl_layout.addWidget(grp)

    def _build_format_group(self):
        grp = QGroupBox("Label Format")
        lay = QVBoxLayout(grp)
        self._fmt_group = QButtonGroup(grp)
        self._radio_json = QRadioButton("LabelMe JSON  (convert to YOLO TXT)")
        self._radio_txt  = QRadioButton("YOLO TXT  (already converted — copy only)")
        self._radio_json.setChecked(True)
        self._fmt_group.addButton(self._radio_json, 0)
        self._fmt_group.addButton(self._radio_txt,  1)
        lay.addWidget(self._radio_json)
        lay.addWidget(self._radio_txt)
        self._ctrl_layout.addWidget(grp)

    def _build_split_group(self):
        grp = QGroupBox("Split Configuration")
        lay = QVBoxLayout(grp)

        ratio_row = QHBoxLayout()
        ratio_row.addWidget(QLabel("Train ratio:"))
        self._slider_ratio = QSlider(Qt.Horizontal)
        self._slider_ratio.setRange(50, 95)
        self._slider_ratio.setValue(80)
        ratio_row.addWidget(self._slider_ratio)
        self._lbl_ratio = QLabel("80% / 20%")
        self._lbl_ratio.setMinimumWidth(75)
        ratio_row.addWidget(self._lbl_ratio)
        lay.addLayout(ratio_row)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel("Random seed:"))
        self._spin_seed = QSpinBox()
        self._spin_seed.setRange(0, 99999)
        self._spin_seed.setValue(42)
        seed_row.addWidget(self._spin_seed)
        seed_row.addStretch()
        lay.addLayout(seed_row)

        self._ctrl_layout.addWidget(grp)

    def _build_output_group(self):
        grp = QGroupBox("Output Folder")
        lay = QVBoxLayout(grp)

        row = QHBoxLayout()
        self._txt_output = QLineEdit()
        self._txt_output.setPlaceholderText("Type or browse a destination folder\u2026")
        self._btn_browse_output = QPushButton("Browse")
        row.addWidget(self._txt_output)
        row.addWidget(self._btn_browse_output)
        lay.addLayout(row)

        self._btn_create_folder = QPushButton("Create Folder")
        lay.addWidget(self._btn_create_folder)

        self._lbl_output_status = QLabel("")
        self._lbl_output_status.setWordWrap(True)
        lay.addWidget(self._lbl_output_status)

        self._ctrl_layout.addWidget(grp)

    def _build_run_group(self):
        grp = QGroupBox("Run")
        lay = QVBoxLayout(grp)
        self._btn_run = QPushButton("Run Split")
        font = self._btn_run.font()
        font.setBold(True)
        self._btn_run.setFont(font)
        lay.addWidget(self._btn_run)
        self._progress = QProgressBar()
        self._progress.setVisible(False)
        lay.addWidget(self._progress)
        self._ctrl_layout.addWidget(grp)

    # ------------------------------------------------------------------
    # Signal wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        self._btn_browse_input.clicked.connect(self._on_browse_input)
        self._btn_scan.clicked.connect(self._on_scan)
        self._slider_ratio.valueChanged.connect(self._on_ratio_changed)
        self._btn_browse_output.clicked.connect(self._on_browse_output)
        self._btn_create_folder.clicked.connect(self._on_create_output_folder)
        self._txt_output.textChanged.connect(self._on_output_path_changed)
        self._btn_run.clicked.connect(self._on_run)
        self._fmt_group.buttonClicked.connect(lambda _: self._on_scan())

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_browse_input(self):
        folder = QFileDialog.getExistingDirectory(
            main_window_parent(self.logger), "Select Input Folder"
        )
        if folder:
            self._txt_input.setText(folder)
            self._on_scan()

    def _on_scan(self):
        folder = self._txt_input.text().strip()
        if not folder or not os.path.isdir(folder):
            self._lbl_scan_result.setText("Invalid folder.")
            return
        src    = Path(folder)
        images = [f for f in src.iterdir()
                  if f.is_file() and f.suffix.lower() in _IMAGE_EXTS]
        use_txt = self._radio_txt.isChecked()
        label_ext = ".txt" if use_txt else ".json"
        label_stems = {f.stem for f in src.iterdir()
                       if f.is_file() and f.suffix.lower() == label_ext}
        paired  = sum(1 for f in images if f.stem in label_stems)
        skipped = len(images) - paired
        label_word = "TXT" if use_txt else "JSON"
        self._lbl_scan_result.setText(
            f"{len(images)} images — {paired} paired with {label_word}, "
            f"{skipped} without {label_word}."
        )
        _set_card(self._card_paired,  str(paired))
        _set_card(self._card_skipped, str(skipped))

    def _on_ratio_changed(self, value: int):
        self._lbl_ratio.setText(f"{value}% / {100 - value}%")

    def _on_browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            main_window_parent(self.logger), "Select Output Folder"
        )
        if folder:
            self._txt_output.setText(folder)

    def _on_output_path_changed(self, text: str):
        """Update status label to reflect whether the typed path exists."""
        path = text.strip()
        if not path:
            self._lbl_output_status.setText("")
        elif os.path.isdir(path):
            self._lbl_output_status.setText("\u2713 Folder exists.")
        else:
            self._lbl_output_status.setText("Folder does not exist — press Create Folder.")

    def _on_create_output_folder(self):
        """Prompt for a parent directory and a new folder name, then create it."""
        parent = self.logger
        dialog = QDialog(main_window_parent(parent))
        dialog.setWindowTitle("Create Output Folder")
        dialog.setMinimumWidth(420)

        layout = QVBoxLayout(dialog)
        form = QFormLayout()

        # Parent directory row
        parent_row = QWidget()
        parent_layout = QHBoxLayout(parent_row)
        parent_layout.setContentsMargins(0, 0, 0, 0)
        self._dlg_txt_parent = QLineEdit()
        self._dlg_txt_parent.setPlaceholderText("Select a parent folder\u2026")
        self._dlg_txt_parent.setText(self._txt_output.text().strip())
        browse_btn = QPushButton("Browse\u2026")
        parent_layout.addWidget(self._dlg_txt_parent)
        parent_layout.addWidget(browse_btn)
        form.addRow("Parent folder:", parent_row)

        # New folder name row
        name_edit = QLineEdit()
        name_edit.setPlaceholderText("e.g. MyDataset_split")
        form.addRow("New folder name:", name_edit)

        layout.addLayout(form)

        # Status label inside dialog
        dlg_status = QLabel("")
        dlg_status.setWordWrap(True)
        layout.addWidget(dlg_status)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.rejected.connect(dialog.reject)

        def _browse_parent():
            folder = QFileDialog.getExistingDirectory(
                dialog, "Select Parent Folder"
            )
            if folder:
                self._dlg_txt_parent.setText(folder)

        def _try_accept():
            p = self._dlg_txt_parent.text().strip()
            n = name_edit.text().strip()
            if not p:
                dlg_status.setText("\u26a0 Please select a parent folder.")
                return
            if not n:
                dlg_status.setText("\u26a0 Please enter a folder name.")
                return
            full = Path(p) / n
            if full.exists():
                dlg_status.setText(f"\u26a0 '{full.name}' already exists in that location.")
                return
            dialog.accept()

        browse_btn.clicked.connect(_browse_parent)
        buttons.accepted.connect(_try_accept)

        if dialog.exec_() != QDialog.Accepted:
            return

        p = self._dlg_txt_parent.text().strip()
        n = name_edit.text().strip()
        new_folder = Path(p) / n

        try:
            for sub in ("train/images", "train/labels", "val/images", "val/labels"):
                (new_folder / sub).mkdir(parents=True, exist_ok=True)
            self._txt_output.setText(str(new_folder))
            self._lbl_output_status.setText(
                f"\u2713 Created: {new_folder}"
            )
            self.logger.info(f"Created output folder structure at: {new_folder}")
        except Exception as exc:
            QMessageBox.critical(
                main_window_parent(self.logger), "Error",
                f"Failed to create folder:\n{exc}"
            )
            self.logger.error(f"Failed to create output folder: {exc}")

    def _on_run(self):
        input_dir  = self._txt_input.text().strip()
        output_dir = self._txt_output.text().strip()

        if not input_dir or not os.path.isdir(input_dir):
            QMessageBox.warning(
                main_window_parent(self.logger), "Missing Input",
                "Please select a valid input folder first."
            )
            return
        if not output_dir:
            QMessageBox.warning(
                main_window_parent(self.logger), "Missing Output",
                "Please select an output folder first."
            )
            return

        # Warn before overwriting
        dst = Path(output_dir)
        existing = [d for d in ("train", "val") if (dst / d).exists()]
        if existing:
            answer = QMessageBox.question(
                main_window_parent(self.logger), "Overwrite Warning",
                f"The following folders already exist in the output directory:\n"
                f"  {', '.join(existing)}\n\nOverwrite them?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return

        train_ratio = self._slider_ratio.value() / 100.0
        seed        = self._spin_seed.value()

        self._txt_log.clear()
        _set_card(self._card_train,   "—")
        _set_card(self._card_val,     "—")
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._btn_run.setEnabled(False)

        if self._radio_txt.isChecked():
            self._worker = _SplitWorkerTxt(input_dir, output_dir, train_ratio, seed)
        else:
            self._worker = _SplitWorker(input_dir, output_dir, train_ratio, seed)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        mode = "YOLO TXT copy" if self._radio_txt.isChecked() else "LabelMe JSON→TXT"
        self.logger.info(f"Dataset split started ({mode}).")

    def _on_progress(self, done: int, total: int):
        self._progress.setMaximum(total)
        self._progress.setValue(done)

    def _on_log(self, message: str):
        self._txt_log.append(message)

    def _on_finished(self, summary: dict):
        self._progress.setVisible(False)
        self._btn_run.setEnabled(True)

        if "error" in summary:
            self.logger.error(f"Split failed: {summary['error']}")
            return

        _set_card(self._card_train,   str(summary.get("train",   "—")))
        _set_card(self._card_val,     str(summary.get("val",     "—")))
        _set_card(self._card_skipped, str(summary.get("skipped", "—")))
        # Write dataset.yaml into the output folder
        output_dir = self._txt_output.text().strip()
        if output_dir:
            class_names = summary.get("class_names", [])
            yaml_content = {
                "path":  output_dir,
                "train": "train/images",
                "val":   "val/images",
                "nc":    len(class_names),
                "names": class_names,
            }
            yaml_path = Path(output_dir) / "dataset.yaml"
            try:
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.dump(yaml_content, f, default_flow_style=False,
                              sort_keys=False, allow_unicode=True)
                self._txt_log.append(f"\ndataset.yaml saved to {yaml_path}")
                self.logger.info(f"dataset.yaml written to {yaml_path}")
            except Exception as exc:
                self.logger.error(f"Failed to write dataset.yaml: {exc}")
        self.logger.info(
            f"Split complete — train: {summary['train']}, "
            f"val: {summary['val']}, skipped: {summary['skipped']}"
        )


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _set_card(card: QGroupBox, value: str):
    card._value_label.setText(value)  # type: ignore[attr-defined]
