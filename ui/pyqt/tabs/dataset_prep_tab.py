"""Dataset Preparation tab – split image+JSON pairs into train/val sets."""

import json
import os
import random
import shutil
from pathlib import Path

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QPushButton, QLineEdit, QSlider, QProgressBar, QScrollArea, QSplitter,
    QFrame, QSpinBox, QTextEdit, QMessageBox, QFileDialog,
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

        self.finished.emit({
            "train":   len(train_stems),
            "val":     len(val_stems),
            "skipped": len(skipped_stems),
            "classes": len(label_to_class),
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
        self._txt_input.setPlaceholderText("Select folder with images + JSON…")
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
        self._txt_output.setReadOnly(True)
        self._txt_output.setPlaceholderText("Select destination folder…")
        self._btn_browse_output = QPushButton("Browse")
        row.addWidget(self._txt_output)
        row.addWidget(self._btn_browse_output)
        lay.addLayout(row)
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
        self._btn_run.clicked.connect(self._on_run)

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
        jsons  = {f.stem for f in src.iterdir()
                  if f.is_file() and f.suffix.lower() == ".json"}
        paired  = sum(1 for f in images if f.stem in jsons)
        skipped = len(images) - paired
        self._lbl_scan_result.setText(
            f"{len(images)} images found — {paired} paired, {skipped} without JSON."
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

        self._worker = _SplitWorker(input_dir, output_dir, train_ratio, seed)
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._on_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.start()
        self.logger.info("Dataset split started.")

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
        self.logger.info(
            f"Split complete — train: {summary['train']}, "
            f"val: {summary['val']}, skipped: {summary['skipped']}"
        )


# ---------------------------------------------------------------------------
# Module-level helper
# ---------------------------------------------------------------------------

def _set_card(card: QGroupBox, value: str):
    card._value_label.setText(value)  # type: ignore[attr-defined]
