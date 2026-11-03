"""Log what happens in the zone to CSV files."""

import csv
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from types import TracebackType

from zone_watch.detection import Detection
from zone_watch.zone_presence import ZonePresence


class _CsvLog:
    """A CSV file that gets a header once, and survives the app closing abruptly."""

    header: tuple[str, ...] = ()

    def __init__(self, path: Path) -> None:
        self.path = path
        self._file = None
        self._writer = None

    def __enter__(self) -> "_CsvLog":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if is_new_file:
            self._writer.writerow(self.header)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._file.close()

    def _write(self, rows: Iterable[Sequence[object]]) -> None:
        """Write rows that start with the current time, and flush them to disk."""
        timestamp = datetime.now().isoformat(timespec="milliseconds")
        self._writer.writerows((timestamp, *row) for row in rows)
        self._file.flush()


class EventLogger(_CsvLog):
    """Append a row whenever a class count changes.

    Logging only the changes keeps the file readable: one row per frame would add
    thousands of identical rows every few minutes.
    """

    header = ("timestamp", "frame", "class", "count")

    def __init__(self, path: Path, class_names: list[str]) -> None:
        super().__init__(path)
        self._counts = {name: 0 for name in class_names}

    def update(self, frame_index: int, counts: dict[str, int]) -> None:
        """Write one row for each class whose count differs from the previous frame."""
        changes = [name for name, count in counts.items() if count != self._counts[name]]
        if changes:
            self._write((frame_index, name, counts[name]) for name in changes)
            self._counts.update(counts)


class ZoneEventLogger(_CsvLog):
    """Append a row whenever a tracked object enters or leaves the zone."""

    header = ("timestamp", "frame", "track_id", "class", "event")

    def __init__(self, path: Path, grace_frames: int) -> None:
        super().__init__(path)
        self._presence = ZonePresence(grace_frames)

    def update(self, frame_index: int, detections: Iterable[Detection]) -> None:
        """Work out this frame's events and write them."""
        events = self._presence.update(frame_index, detections)
        if events:
            self._write((e.frame_index, e.track_id, e.class_name, e.event) for e in events)
