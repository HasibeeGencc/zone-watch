"""The main loop: read a frame, detect, filter by zone, draw, log and display."""

import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from zone_watch.config import AppConfig
from zone_watch.detection import Detector, ZoneState, build_zone_state, filter_in_roi
from zone_watch.events import EventLogger, ZoneEventLogger
from zone_watch.overlay import draw_frame
from zone_watch.roi import Roi, RoiSelector

WINDOW_NAME = "Zone Watch"
HINT = "Drag with the mouse to draw a zone | r: reset | q: quit"
DEFAULT_VIDEO_FPS = 30.0
FPS_SMOOTHING = 0.9


@dataclass(frozen=True)
class RunOptions:
    """Settings that describe one particular run."""

    source: int | str
    initial_roi: Roi | None
    display: bool
    save_video: Path | None


class FpsMeter:
    """Measure frames per second, smoothed so that the number on screen is readable."""

    def __init__(self) -> None:
        self.fps = 0.0
        self._last_tick: float | None = None

    def tick(self) -> None:
        """Record that a frame has finished."""
        now = time.perf_counter()
        if self._last_tick is not None:
            current = 1 / max(now - self._last_tick, 1e-6)
            self.fps = current if self.fps == 0 else FPS_SMOOTHING * self.fps + (1 - FPS_SMOOTHING) * current
        self._last_tick = now

    def restart(self) -> None:
        """Ignore the time since the last frame, such as a pause while drawing a zone."""
        self._last_tick = None


def open_source(source: int | str) -> cv2.VideoCapture:
    """Open a webcam or video file, stopping with a clear message if that fails."""
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise SystemExit(f"Could not open source: {source}")
    return capture


def create_video_writer(path: Path, capture: cv2.VideoCapture, frame: np.ndarray) -> cv2.VideoWriter:
    """Create the output video at the source's frame rate and the frame's size."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_VIDEO_FPS
    height, width = frame.shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Could not create output video: {path}")
    return writer


def analyse_frame(frame: np.ndarray, roi: Roi | None, detector: Detector) -> ZoneState:
    """Detect objects and keep those in the zone; without a zone there is nothing to do."""
    if roi is None:
        return build_zone_state([], detector.class_names, detector.config.alert_class)
    detections = filter_in_roi(detector.detect(frame), roi)
    return build_zone_state(detections, detector.class_names, detector.config.alert_class)


def window_was_closed() -> bool:
    """Return whether the user closed the window with its X button."""
    return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1


def fit_scale(width: int, height: int, max_width: int, max_height: int) -> float:
    """Return the factor that fits a frame in the given size, never enlarging it."""
    return min(1.0, max_width / width, max_height / height)


def open_window(selector: RoiSelector, scale: float) -> None:
    """Create the window and send it mouse events in frame coordinates."""

    def on_mouse(event: int, x: int, y: int, *_: object) -> None:
        # The window shows a shrunken frame, but the zone is in original pixels.
        selector.handle_mouse(event, round(x / scale), round(y / scale))

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)


def show_frame(canvas: np.ndarray, scale: float) -> None:
    """Show the frame in the window, shrunk to fit if needed."""
    if scale < 1.0:
        canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    cv2.imshow(WINDOW_NAME, canvas)


def run(config: AppConfig, options: RunOptions) -> None:
    """Open the source and process frames until it ends or the user quits."""
    detector = Detector(config)
    capture = open_source(options.source)
    writer = None
    try:
        ok, frame = capture.read()
        if not ok:
            raise SystemExit(f"Could not read a frame from source: {options.source}")

        height, width = frame.shape[:2]
        roi = options.initial_roi.clipped(width, height) if options.initial_roi else None
        if roi is not None and not roi.is_large_enough():
            raise SystemExit(f"The zone is too small or outside the {width}x{height} frame: {roi}")
        selector = RoiSelector(width, height, roi)

        scale = fit_scale(width, height, config.max_display_width, config.max_display_height)
        if options.display:
            open_window(selector, scale)
        if options.save_video:
            writer = create_video_writer(options.save_video, capture, frame)

        zone_log = (
            ZoneEventLogger(config.zone_events_path, config.leave_grace_frames)
            if config.tracking
            else nullcontext()
        )
        with EventLogger(config.log_path, detector.class_names) as logger, zone_log as zone_logger:
            frames_done = process_frames(
                capture, frame, selector, detector, logger, zone_logger, writer, options, scale
            )
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if options.display:
            cv2.destroyAllWindows()

    print(f"Processed {frames_done} frames. Events logged to {config.log_path}")
    if config.tracking:
        print(f"Zone entries and exits logged to {config.zone_events_path}")
    if options.save_video:
        print(f"Annotated video saved to {options.save_video}")


def process_frames(
    capture: cv2.VideoCapture,
    first_frame: np.ndarray,
    selector: RoiSelector,
    detector: Detector,
    logger: EventLogger,
    zone_logger: ZoneEventLogger | None,
    writer: cv2.VideoWriter | None,
    options: RunOptions,
    display_scale: float,
) -> int:
    """Run the frame loop and return how many frames were processed."""
    # A video file waits on its current frame until a zone exists, otherwise it would
    # race ahead while the user draws. A webcam is live, so it keeps running.
    is_video_file = isinstance(options.source, str)
    frame = first_frame
    frame_index = 0
    meter = FpsMeter()

    while True:
        state = analyse_frame(frame, selector.roi, detector)
        canvas = draw_frame(
            frame, selector.roi, selector.preview, state, detector.config.alert_class, meter.fps, HINT
        )
        waiting_for_zone = is_video_file and selector.roi is None

        if not waiting_for_zone:
            logger.update(frame_index, state.counts)
            if zone_logger is not None:
                zone_logger.update(frame_index, state.detections)
            if writer is not None:
                writer.write(canvas)
            frame_index += 1
            meter.tick()
        else:
            meter.restart()

        if options.display:
            show_frame(canvas, display_scale)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or window_was_closed():
                break
            if key == ord("r"):
                selector.reset()

        if not waiting_for_zone:
            ok, frame = capture.read()
            if not ok:
                break

    return frame_index
