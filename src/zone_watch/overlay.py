"""Draw the zone, boxes, counts and alert on a frame."""

import cv2
import numpy as np

from zone_watch.detection import Detection, ZoneState
from zone_watch.roi import Roi

# OpenCV colours are in blue, green, red order.
GREEN = (0, 200, 0)
RED = (0, 0, 255)
YELLOW = (0, 220, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

FONT = cv2.FONT_HERSHEY_SIMPLEX
ROI_THICKNESS = 3
LINE_HEIGHT = 26


def draw_text(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int] = WHITE,
    background: tuple[int, int, int] = BLACK,
    scale: float = 0.6,
) -> None:
    """Draw text on a filled rectangle so that it stays readable on any image."""
    x, y = position
    (width, height), baseline = cv2.getTextSize(text, FONT, scale, 1)
    cv2.rectangle(frame, (x, y - height - 4), (x + width + 6, y + baseline), background, cv2.FILLED)
    cv2.putText(frame, text, (x + 3, y - 2), FONT, scale, color, 1, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, detections: list[Detection], alert_class: str) -> None:
    """Draw each box with its class and confidence, using red for the alert class."""
    for detection in detections:
        color = RED if detection.class_name == alert_class else GREEN
        top_left = int(detection.x1), int(detection.y1)
        bottom_right = int(detection.x2), int(detection.y2)
        cv2.rectangle(frame, top_left, bottom_right, color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f}"
        # Keep the label inside the frame when a box touches the top edge.
        label_position = top_left[0], max(top_left[1] - 2, 20)
        draw_text(frame, label, label_position, WHITE, color, scale=0.5)


def draw_roi(frame: np.ndarray, roi: Roi, alert: bool) -> None:
    """Draw the zone border, which turns red during an alert."""
    color = RED if alert else GREEN
    cv2.rectangle(frame, (roi.x1, roi.y1), (roi.x2, roi.y2), color, ROI_THICKNESS)


def draw_preview(frame: np.ndarray, roi: Roi) -> None:
    """Draw the rectangle that is currently being dragged."""
    cv2.rectangle(frame, (roi.x1, roi.y1), (roi.x2, roi.y2), YELLOW, 2)


def draw_status(frame: np.ndarray, counts: dict[str, int], fps: float) -> None:
    """Draw the per-class counts and the FPS in the top-left corner."""
    lines = [f"{name} in zone: {count}" for name, count in counts.items()]
    lines.append(f"FPS: {fps:.1f}")
    for index, line in enumerate(lines, start=1):
        draw_text(frame, line, (10, index * LINE_HEIGHT))


def draw_alert(frame: np.ndarray, alert_class: str, count: int) -> None:
    """Draw the alert banner along the bottom of the frame."""
    text = f"ALERT: {alert_class} in zone ({count})"
    draw_text(frame, text, (10, frame.shape[0] - 15), WHITE, RED, scale=0.8)


def draw_hint(frame: np.ndarray, text: str) -> None:
    """Draw a short instruction along the bottom of the frame."""
    draw_text(frame, text, (10, frame.shape[0] - 15), BLACK, YELLOW)


def draw_frame(
    frame: np.ndarray,
    roi: Roi | None,
    preview: Roi | None,
    state: ZoneState,
    alert_class: str,
    fps: float,
    hint: str,
) -> np.ndarray:
    """Return a copy of the frame with everything drawn on it."""
    canvas = frame.copy()
    draw_detections(canvas, state.detections, alert_class)
    if roi is not None:
        draw_roi(canvas, roi, state.alert)
    if preview is not None:
        draw_preview(canvas, preview)
    draw_status(canvas, state.counts, fps)
    if state.alert:
        draw_alert(canvas, alert_class, state.counts[alert_class])
    elif roi is None:
        draw_hint(canvas, hint)
    return canvas
