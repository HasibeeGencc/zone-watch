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
BOX_THICKNESS = 2
LINE_HEIGHT = 26

# Sizes below are for a frame this tall, and grow with taller frames. A fixed text
# size would be unreadable on a 4K video.
REFERENCE_HEIGHT = 720


def ui_scale(frame: np.ndarray) -> float:
    """Return how much larger to draw text and lines on this frame."""
    return max(frame.shape[0] / REFERENCE_HEIGHT, 1.0)


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
    ui = ui_scale(frame)
    font_scale = scale * ui
    thickness = max(1, round(ui))
    padding = round(4 * ui)
    (width, height), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    cv2.rectangle(frame, (x, y - height - padding), (x + width + padding, y + baseline), background, cv2.FILLED)
    cv2.putText(frame, text, (x + padding // 2, y - padding // 2), FONT, font_scale, color, thickness, cv2.LINE_AA)


def draw_detections(frame: np.ndarray, detections: list[Detection], alert_class: str) -> None:
    """Draw each box with its class and confidence, using red for the alert class."""
    ui = ui_scale(frame)
    for detection in detections:
        color = RED if detection.class_name == alert_class else GREEN
        top_left = int(detection.x1), int(detection.y1)
        bottom_right = int(detection.x2), int(detection.y2)
        cv2.rectangle(frame, top_left, bottom_right, color, round(BOX_THICKNESS * ui))
        label = f"{detection.class_name} {detection.confidence:.2f}"
        # Keep the label inside the frame when a box touches the top edge.
        label_position = top_left[0], max(top_left[1] - round(2 * ui), round(20 * ui))
        draw_text(frame, label, label_position, WHITE, color, scale=0.5)


def draw_roi(frame: np.ndarray, roi: Roi, alert: bool) -> None:
    """Draw the zone border, which turns red during an alert."""
    color = RED if alert else GREEN
    thickness = round(ROI_THICKNESS * ui_scale(frame))
    cv2.rectangle(frame, (roi.x1, roi.y1), (roi.x2, roi.y2), color, thickness)


def draw_preview(frame: np.ndarray, roi: Roi) -> None:
    """Draw the rectangle that is currently being dragged."""
    thickness = round(BOX_THICKNESS * ui_scale(frame))
    cv2.rectangle(frame, (roi.x1, roi.y1), (roi.x2, roi.y2), YELLOW, thickness)


def draw_status(frame: np.ndarray, counts: dict[str, int], fps: float) -> None:
    """Draw the per-class counts and the FPS in the top-left corner."""
    ui = ui_scale(frame)
    lines = [f"{name} in zone: {count}" for name, count in counts.items()]
    lines.append(f"FPS: {fps:.1f}")
    for index, line in enumerate(lines, start=1):
        draw_text(frame, line, (round(10 * ui), round(index * LINE_HEIGHT * ui)))


def bottom_position(frame: np.ndarray) -> tuple[int, int]:
    """Return where a message along the bottom of the frame starts."""
    ui = ui_scale(frame)
    return round(10 * ui), frame.shape[0] - round(15 * ui)


def draw_alert(frame: np.ndarray, alert_class: str, count: int) -> None:
    """Draw the alert banner along the bottom of the frame."""
    text = f"ALERT: {alert_class} in zone ({count})"
    draw_text(frame, text, bottom_position(frame), WHITE, RED, scale=0.8)


def draw_hint(frame: np.ndarray, text: str) -> None:
    """Draw a short instruction along the bottom of the frame."""
    draw_text(frame, text, bottom_position(frame), BLACK, YELLOW)


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
