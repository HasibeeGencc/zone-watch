"""The region of interest (ROI) and the mouse logic for drawing it."""

from dataclasses import dataclass

import cv2

# Smaller rectangles are treated as accidental clicks rather than zones.
MIN_ROI_SIZE = 10


@dataclass(frozen=True)
class Roi:
    """A rectangle in pixel coordinates, with (x1, y1) as its top-left corner."""

    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def from_corners(cls, corner_a: tuple[int, int], corner_b: tuple[int, int]) -> "Roi":
        """Build a ROI from two opposite corners, in whichever direction it was dragged."""
        (ax, ay), (bx, by) = corner_a, corner_b
        return cls(min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))

    def contains(self, x: float, y: float) -> bool:
        """Return whether a point is inside the ROI or on its border."""
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def clipped(self, width: int, height: int) -> "Roi":
        """Return this ROI limited to the frame, as a drag can end outside the window."""
        return Roi(
            max(0, min(self.x1, width - 1)),
            max(0, min(self.y1, height - 1)),
            max(0, min(self.x2, width - 1)),
            max(0, min(self.y2, height - 1)),
        )

    def is_large_enough(self) -> bool:
        """Return whether both sides reach the minimum zone size."""
        return self.x2 - self.x1 >= MIN_ROI_SIZE and self.y2 - self.y1 >= MIN_ROI_SIZE


class RoiSelector:
    """Turn click-and-drag mouse events into a ROI."""

    def __init__(self, frame_width: int, frame_height: int, roi: Roi | None = None) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.roi = roi
        self._drag_start: tuple[int, int] | None = None
        self._drag_end: tuple[int, int] | None = None

    @property
    def preview(self) -> Roi | None:
        """Return the rectangle being dragged right now, if any."""
        if self._drag_start is None or self._drag_end is None:
            return None
        return self._rectangle()

    def handle_mouse(self, event: int, x: int, y: int, *_: object) -> None:
        """Handle an OpenCV mouse event; the signature is what cv2.setMouseCallback expects."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = self._drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_start is not None:
            self._drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and self._drag_start is not None:
            self._drag_end = (x, y)
            rectangle = self._rectangle()
            # An accidental click keeps the current zone instead of erasing it.
            if rectangle.is_large_enough():
                self.roi = rectangle
            self._drag_start = self._drag_end = None

    def reset(self) -> None:
        """Clear the zone and any drag in progress."""
        self.roi = None
        self._drag_start = self._drag_end = None

    def _rectangle(self) -> Roi:
        """Return the dragged rectangle, limited to the frame."""
        rectangle = Roi.from_corners(self._drag_start, self._drag_end)
        return rectangle.clipped(self.frame_width, self.frame_height)
