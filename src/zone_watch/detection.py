"""Run the YOLO model and work out what is inside the zone."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from zone_watch.config import AppConfig
from zone_watch.roi import Roi


@dataclass(frozen=True)
class Detection:
    """One box found by the model, in pixel coordinates."""

    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def centre(self) -> tuple[float, float]:
        """Return the middle point of the box."""
        return (self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2


@dataclass(frozen=True)
class ZoneState:
    """What is inside the zone in one frame."""

    detections: list[Detection]
    counts: dict[str, int]
    alert: bool


class Detector:
    """Wrap the Ultralytics model behind a small interface."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.model = self._load_model(config.model_path)
        self.class_names = [self.model.names[i] for i in sorted(self.model.names)]
        if config.alert_class not in self.class_names:
            raise SystemExit(
                f"alert_class '{config.alert_class}' is not in the model's classes: {self.class_names}"
            )

    @staticmethod
    def _load_model(path: Path) -> YOLO:
        """Load the weights, stopping with a clear message if the file is missing."""
        if not path.exists():
            raise SystemExit(f"Model file not found: {path}")
        return YOLO(str(path))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return every box the model finds in a BGR frame."""
        result = self.model.predict(
            frame,
            conf=self.config.confidence,
            imgsz=self.config.image_size,
            device=self.config.device,
            # One head should count once: without this, overlapping boxes of different
            # classes both survive and one person can raise a false alert.
            agnostic_nms=True,
            verbose=False,
        )[0]
        boxes = result.boxes
        return [
            Detection(self.model.names[int(class_id)], float(confidence), *map(float, corners))
            for class_id, confidence, corners in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist()
            )
        ]


def filter_in_roi(detections: Iterable[Detection], roi: Roi) -> list[Detection]:
    """Keep only the detections whose centre is inside the zone."""
    return [detection for detection in detections if roi.contains(*detection.centre)]


def build_zone_state(
    detections: list[Detection], class_names: list[str], alert_class: str
) -> ZoneState:
    """Count the zone's detections per class and decide whether to raise the alert."""
    counts = {name: 0 for name in class_names}
    for detection in detections:
        counts[detection.class_name] += 1
    return ZoneState(detections, counts, counts[alert_class] > 0)
