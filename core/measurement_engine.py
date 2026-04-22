"""Measurement engine for YOLO edge-detection results.

Extracts Y-difference calculations, calibration, and judgment logic
so it can be reused across UI tabs, FastAPI endpoints, or CLI scripts.
"""


class MeasurementEngine:
    """Calculates edge measurements and judgments from YOLO detection results."""

    def __init__(
        self,
        microns_per_pixel=2.3,
        block1_offset=0.0,
        block2_offset=0.0,
        measurement_offset_microns=0.0,
        judgment_criteria=None,
    ):
        self.microns_per_pixel = microns_per_pixel
        self.block1_offset = block1_offset
        self.block2_offset = block2_offset
        self.measurement_offset_microns = measurement_offset_microns
        self.judgment_criteria = judgment_criteria or {"good": 10, "acceptable": 20}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self, boxes_xyxy, boxes_xywh, classes, class_name_fn):
        """Calculate Y-difference and judgment from detection arrays.

        Args:
            boxes_xyxy: ndarray of shape (N, 4) – [x1, y1, x2, y2].
            boxes_xywh: ndarray of shape (N, 4) – [cx, cy, w, h].
            classes: ndarray of shape (N,) – integer class IDs.
            class_name_fn: callable(int) -> str  mapping class ID to name.

        Returns:
            dict with keys:
                y_diff_microns  – float or None
                judgment         – str or None  ("Good" / "Acceptable" / "No Good")
                microns_per_pixel – float (calibrated or default)
                block1_edge_y   – float or None
                block2_edge_y   – float or None
        """
        microns_per_pixel = self._calibrate(boxes_xywh, classes, class_name_fn)
        block1_edge_y, block2_edge_y = self._find_edges(
            boxes_xywh, classes, class_name_fn, microns_per_pixel
        )

        y_diff_microns = None
        judgment = None

        if block1_edge_y is not None and block2_edge_y is not None:
            y_diff_pixels = block1_edge_y - block2_edge_y
            y_diff_microns = (y_diff_pixels * microns_per_pixel) + self.measurement_offset_microns
            judgment = self._judge(y_diff_microns)

        return {
            "y_diff_microns": y_diff_microns,
            "judgment": judgment,
            "microns_per_pixel": microns_per_pixel,
            "block1_edge_y": block1_edge_y,
            "block2_edge_y": block2_edge_y,
        }

    def calculate_from_results(self, results, class_name_fn):
        """Convenience wrapper that accepts a YOLO `results` object directly.

        Args:
            results: single YOLO Results object (e.g. ``results[0]``).
            class_name_fn: callable(int) -> str.

        Returns:
            Same dict as :meth:`calculate`, or a dict with all-None values
            when boxes are empty.
        """
        if not hasattr(results.boxes, "xyxy") or len(results.boxes.xyxy) == 0:
            return {
                "y_diff_microns": None,
                "judgment": None,
                "microns_per_pixel": self.microns_per_pixel,
                "block1_edge_y": None,
                "block2_edge_y": None,
            }

        boxes_xyxy = results.boxes.xyxy.cpu().numpy()
        boxes_xywh = results.boxes.xywh.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        return self.calculate(boxes_xyxy, boxes_xywh, classes, class_name_fn)

    def judge(self, y_diff_microns):
        """Public alias for judgment logic."""
        return self._judge(y_diff_microns)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calibrate(self, boxes_xywh, classes, class_name_fn):
        """Return microns-per-pixel using cal_mark if present, else default."""
        for box_xywh, cls in zip(boxes_xywh, classes):
            if class_name_fn(int(cls)) == "cal_mark":
                width = float(box_xywh[2])
                if width > 0:
                    return 1000.0 / width
        return self.microns_per_pixel

    def _find_edges(self, boxes_xywh, classes, class_name_fn, microns_per_pixel):
        block1_edge_y = None
        block2_edge_y = None

        for box_xywh, cls in zip(boxes_xywh, classes):
            _, y_center, _, height = box_xywh
            name = class_name_fn(int(cls))
            edge_y = float(y_center) + float(height) / 2

            if name in ("block1_edge", "block1_edge15"):
                block1_edge_y = edge_y + (self.block1_offset / microns_per_pixel)
            elif name in ("block2_edge", "block2_edge15"):
                block2_edge_y = edge_y + (self.block2_offset / microns_per_pixel)

        return block1_edge_y, block2_edge_y

    def _judge(self, y_diff_microns):
        if y_diff_microns < self.judgment_criteria["good"]:
            return "Good"
        elif y_diff_microns < self.judgment_criteria["acceptable"]:
            return "Acceptable"
        return "No Good"
