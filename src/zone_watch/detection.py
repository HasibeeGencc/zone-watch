"""Run the YOLO model and work out what is inside the zone."""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from zone_watch.config import AppConfig
from zone_watch.roi import Roi

# Matches track_low_thresh in configs/bytetrack.yaml.
TRACKER_MIN_CONFIDENCE = 0.1


@dataclass(frozen=True)
class Detection:
    """One box found by the model, in pixel coordinates."""

    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: int | None = None

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
        """Return every box the model finds in a BGR frame, with IDs if tracking is on."""
        options = {
            "imgsz": self.config.image_size,
            "device": self.config.device,
            # One head should count once: without this, overlapping boxes of different
            # classes both survive and one person can raise a false alert.
            "agnostic_nms": True,
            "verbose": False,
        }
        if self.config.tracking:
            # ByteTrack needs the weak boxes too, and applies its own thresholds to them.
            result = self.model.track(
                frame, conf=TRACKER_MIN_CONFIDENCE, tracker=str(self.config.tracker_config),
                persist=True, **options,
            )[0]
        else:
            result = self.model.predict(frame, conf=self.config.confidence, **options)[0]

        boxes = result.boxes
        # The tracker leaves the IDs empty when a frame has no tracked objects.
        track_ids = boxes.id.tolist() if boxes.id is not None else [None] * len(boxes)
        return [
            Detection(
                self.model.names[int(class_id)], float(confidence), *map(float, corners),
                track_id=None if track_id is None else int(track_id),
            )
            for class_id, confidence, corners, track_id in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xyxy.tolist(), track_ids
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
