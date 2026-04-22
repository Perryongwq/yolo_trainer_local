"""Reusable OpenCV / Qt image rendering helpers.

Provides drawing functions for bounding boxes, edge lines, measurements,
and Qt pixmap conversion — shared between evaluation and annotation tabs.
"""
import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt


# ---------------------------------------------------------------------------
# OpenCV drawing
# ---------------------------------------------------------------------------

def draw_detection_boxes(img_rgb, boxes_xyxy, classes, confidences, class_name_fn,
                         show_labels=True, show_conf=True):
    """Draw bounding boxes with optional labels/confidence on an RGB image.

    Args:
        img_rgb: numpy HWC RGB image (modified in-place).
        boxes_xyxy: ndarray (N,4) pixel coords.
        classes: ndarray (N,) int class ids.
        confidences: ndarray (N,) floats.
        class_name_fn: callable(int)->str.
        show_labels: draw class name.
        show_conf: draw confidence value.
    """
    for box, cls, conf in zip(boxes_xyxy, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        class_name = class_name_fn(int(cls))
        color = (0, 255, 0)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)

        parts = []
        if show_labels:
            parts.append(class_name)
        if show_conf:
            parts.append(f"{conf:.2f}")

        if parts:
            _draw_label(img_rgb, " ".join(parts), x1, y1, color)


def draw_annotation_boxes(img_rgb, annotations, class_name_fn):
    """Draw annotation dicts (from AnnotationManager) on an RGB image.

    Each annotation: {'bbox': [x1,y1,x2,y2], 'class_id': int, 'confidence': float}
    """
    for anno in annotations:
        bbox = anno["bbox"]
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        class_name = class_name_fn(anno["class_id"])
        conf = anno.get("confidence", 1.0)
        color = (0, 255, 0)

        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)
        _draw_label(img_rgb, f"{class_name} {conf:.2f}", x1, y1, color)


def draw_edge_measurements(img_rgb, boxes_xyxy, boxes_xywh, classes, confidences,
                           class_name_fn, measurement_result,
                           show_labels=True, show_conf=True, show_measurements=True,
                           microns_per_pixel=2.3):
    """Draw edge-detection annotations (lines, calibration, judgment) on an RGB image.

    This is the evaluation-tab-specific renderer that handles block edges,
    calibration markers, and Y-difference text.

    Args:
        measurement_result: dict returned by MeasurementEngine.calculate().
    """
    h, w = img_rgb.shape[:2]
    mpp = measurement_result.get("microns_per_pixel", microns_per_pixel)

    for box, box_xywh, cls, conf in zip(boxes_xyxy, boxes_xywh, classes, confidences):
        x1, y1, x2, y2 = map(int, box)
        x_center, y_center, width_box, height_box = box_xywh
        class_name = class_name_fn(int(cls))
        x_center_i = int(x_center)
        y_center_i = int(y_center)
        height_i = int(height_box)

        if class_name in ("block1_edge", "block1_edge15"):
            edge_y = y_center_i + height_i // 2
            cv2.line(img_rgb, (x_center_i - 300, edge_y), (x_center_i + 300, edge_y), (255, 0, 0), 2)
            label_x, label_y, color, is_edge = x_center_i + 300, edge_y, (255, 0, 0), True
        elif class_name in ("block2_edge", "block2_edge15"):
            edge_y = y_center_i + height_i // 2
            cv2.line(img_rgb, (x_center_i - 300, edge_y), (x_center_i + 300, edge_y), (0, 255, 255), 2)
            label_x, label_y, color, is_edge = x_center_i + 300, edge_y, (0, 255, 255), True
        elif class_name == "cal_mark":
            continue  # calibration marker – don't draw
        elif class_name in ("block1", "block1_15", "block2", "block2_15"):
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_x, label_y, color, is_edge = x1, y1, (0, 255, 0), False
        else:
            cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_x, label_y, color, is_edge = x1, y1, (0, 255, 0), False

        # Label
        parts = []
        if show_labels:
            parts.append(class_name)
        if show_conf:
            parts.append(f"{conf:.2f}")
        if parts:
            _draw_label(img_rgb, " ".join(parts), label_x, label_y, color, font_scale=0.9)

        # Measurements per box
        if show_measurements:
            if is_edge:
                meas_text = f"Edge Y: {label_y}px"
                cv2.putText(img_rgb, meas_text, (x_center_i - 80, label_y + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            else:
                w_px, h_px = x2 - x1, y2 - y1
                meas_text = f"W:{w_px * mpp:.1f}\u03bcm H:{h_px * mpp:.1f}\u03bcm"
                cv2.putText(img_rgb, meas_text, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Calibration info
    cv2.putText(img_rgb, f"{mpp:.2f} um/pixel", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Y-difference & judgment
    b1 = measurement_result.get("block1_edge_y")
    b2 = measurement_result.get("block2_edge_y")
    y_diff = measurement_result.get("y_diff_microns")
    judgment = measurement_result.get("judgment")

    if b1 is not None and b2 is not None and y_diff is not None:
        text_x = w // 2 + 250
        text_y = int((b1 + b2) / 2)

        cv2.putText(img_rgb, f"{y_diff:.2f} microns", (text_x - 100, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        jcolors = {"Good": (0, 255, 0), "Acceptable": (0, 165, 255), "No Good": (0, 0, 255)}
        jcolor = jcolors.get(judgment, (255, 255, 255))
        cv2.putText(img_rgb, f"Judgment: {judgment}", (text_x - 100, text_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, jcolor, 2)

        from datetime import datetime
        cv2.putText(img_rgb, f"Checked on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)


# ---------------------------------------------------------------------------
# Qt helpers
# ---------------------------------------------------------------------------

def cv_to_pixmap(img_rgb):
    """Convert an RGB numpy image to QPixmap."""
    h, w, ch = img_rgb.shape
    bytes_per_line = 3 * w
    q_image = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(q_image)


def load_pixmap_scaled(image_path, scale=0.5):
    """Load an image file into a QPixmap scaled by *scale* (default 50%)."""
    pixmap = QPixmap(image_path)
    if pixmap.isNull():
        return None
    return pixmap.scaled(
        int(pixmap.width() * scale),
        int(pixmap.height() * scale),
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )


def fit_pixmap(pixmap, max_width, max_height):
    """Scale a QPixmap to fit within max_width × max_height, keeping aspect ratio."""
    if pixmap.isNull():
        return pixmap
    return pixmap.scaled(max_width, max_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------

def _draw_label(img, text, x, y, color, font_scale=0.7, thickness=2):
    """Draw a text label with background rectangle at (x, y-offset)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x, y - th - baseline - 5), (x + tw + 5, y), color, -1)
    cv2.putText(img, text, (x + 2, y - baseline - 2), font, font_scale, (255, 255, 255), thickness)
