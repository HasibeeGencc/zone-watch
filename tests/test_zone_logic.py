"""Tests for the logic that does not need a model or a camera."""

import csv

import cv2

from zone_watch.detection import Detection, build_zone_state, filter_in_roi
from zone_watch.events import EventLogger
from zone_watch.roi import Roi, RoiSelector

CLASSES = ["With Helmet", "Without Helmet"]
ALERT_CLASS = "Without Helmet"


def make_detection(class_name: str, x1: float, y1: float, x2: float, y2: float) -> Detection:
    return Detection(class_name, 0.9, x1, y1, x2, y2)


def drag(selector: RoiSelector, start: tuple[int, int], end: tuple[int, int]) -> None:
    selector.handle_mouse(cv2.EVENT_LBUTTONDOWN, *start)
    selector.handle_mouse(cv2.EVENT_MOUSEMOVE, *end)
    selector.handle_mouse(cv2.EVENT_LBUTTONUP, *end)


def test_roi_is_normalised_when_dragged_backwards() -> None:
    assert Roi.from_corners((100, 80), (20, 10)) == Roi(20, 10, 100, 80)


def test_roi_contains_points_on_its_border_only_up_to_the_edge() -> None:
    roi = Roi(10, 10, 20, 20)
    assert roi.contains(10, 20)
    assert not roi.contains(9.9, 15)
    assert not roi.contains(15, 20.1)


def test_selector_clips_a_drag_that_ends_outside_the_frame() -> None:
    selector = RoiSelector(640, 480)
    drag(selector, (600, 400), (900, -50))
    assert selector.roi == Roi(600, 0, 639, 400)


def test_selector_shows_a_preview_only_during_the_drag() -> None:
    selector = RoiSelector(640, 480)
    selector.handle_mouse(cv2.EVENT_LBUTTONDOWN, 10, 10)
    selector.handle_mouse(cv2.EVENT_MOUSEMOVE, 100, 100)
    assert selector.preview == Roi(10, 10, 100, 100)
    assert selector.roi is None

    selector.handle_mouse(cv2.EVENT_LBUTTONUP, 100, 100)
    assert selector.preview is None
    assert selector.roi == Roi(10, 10, 100, 100)


def test_accidental_click_keeps_the_current_zone() -> None:
    selector = RoiSelector(640, 480, Roi(0, 0, 100, 100))
    drag(selector, (50, 50), (52, 52))
    assert selector.roi == Roi(0, 0, 100, 100)


def test_mouse_move_without_a_click_does_nothing() -> None:
    selector = RoiSelector(640, 480)
    selector.handle_mouse(cv2.EVENT_MOUSEMOVE, 100, 100)
    assert selector.preview is None
    assert selector.roi is None


def test_reset_clears_the_zone() -> None:
    selector = RoiSelector(640, 480, Roi(0, 0, 100, 100))
    selector.reset()
    assert selector.roi is None


def test_filter_uses_the_box_centre_not_its_overlap() -> None:
    roi = Roi(100, 100, 200, 200)
    centre_inside = make_detection("With Helmet", 50, 100, 240, 200)
    overlaps_but_centre_outside = make_detection("With Helmet", 150, 150, 400, 400)
    assert filter_in_roi([centre_inside, overlaps_but_centre_outside], roi) == [centre_inside]


def test_alert_is_raised_only_for_the_alert_class() -> None:
    helmet = make_detection("With Helmet", 0, 0, 10, 10)
    no_helmet = make_detection(ALERT_CLASS, 0, 0, 10, 10)

    assert not build_zone_state([helmet], CLASSES, ALERT_CLASS).alert

    state = build_zone_state([helmet, no_helmet, no_helmet], CLASSES, ALERT_CLASS)
    assert state.alert
    assert state.counts == {"With Helmet": 1, ALERT_CLASS: 2}


def test_logger_writes_only_changes(tmp_path) -> None:
    path = tmp_path / "logs" / "events.csv"
    with EventLogger(path, CLASSES) as logger:
        logger.update(0, {"With Helmet": 0, ALERT_CLASS: 0})
        logger.update(1, {"With Helmet": 2, ALERT_CLASS: 0})
        logger.update(2, {"With Helmet": 2, ALERT_CLASS: 0})
        logger.update(3, {"With Helmet": 1, ALERT_CLASS: 1})

    with path.open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    logged = [(row["frame"], row["class"], row["count"]) for row in rows]
    assert logged == [
        ("1", "With Helmet", "2"),
        ("3", "With Helmet", "1"),
        ("3", ALERT_CLASS, "1"),
    ]


def test_logger_appends_without_repeating_the_header(tmp_path) -> None:
    path = tmp_path / "events.csv"
    for frame in (1, 2):
        with EventLogger(path, CLASSES) as logger:
            logger.update(frame, {"With Helmet": 1, ALERT_CLASS: 0})

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "timestamp,frame,class,count"
    assert len(lines) == 3
