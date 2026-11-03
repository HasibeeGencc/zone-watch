"""Work out when tracked objects enter and leave the zone."""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from zone_watch.detection import Detection

ENTERED = "entered"
LEFT = "left"


@dataclass(frozen=True)
class ZoneEvent:
    """One object entering or leaving the zone."""

    frame_index: int
    track_id: int
    class_name: str
    event: str


@dataclass
class _TrackedObject:
    """What is known about an object currently in the zone."""

    last_seen: int
    class_votes: Counter = field(default_factory=Counter)


class ZonePresence:
    """Turn the tracked detections in the zone into entered and left events.

    A detector misses objects now and then, so an object only counts as gone after
    being unseen for more than `grace_frames`. Otherwise each brief miss would log a
    false left and entered pair.
    """

    def __init__(self, grace_frames: int) -> None:
        self.grace_frames = grace_frames
        self._objects: dict[int, _TrackedObject] = {}

    def update(self, frame_index: int, detections: Iterable[Detection]) -> list[ZoneEvent]:
        """Record one frame's detections in the zone and return the resulting events."""
        events = []
        for detection in detections:
            # A box that the tracker has not confirmed yet has no ID to follow.
            if detection.track_id is None:
                continue
            tracked = self._objects.get(detection.track_id)
            if tracked is None:
                tracked = self._objects[detection.track_id] = _TrackedObject(frame_index)
                events.append(ZoneEvent(frame_index, detection.track_id, detection.class_name, ENTERED))
            tracked.last_seen = frame_index
            tracked.class_votes[detection.class_name] += 1

        for track_id, tracked in list(self._objects.items()):
            if frame_index - tracked.last_seen > self.grace_frames:
                # The class can flicker between frames, so report the most common one.
                class_name = tracked.class_votes.most_common(1)[0][0]
                events.append(ZoneEvent(frame_index, track_id, class_name, LEFT))
                del self._objects[track_id]
        return events
