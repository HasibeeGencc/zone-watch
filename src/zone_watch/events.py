"""Log changes in the zone's counts to a CSV file."""

import csv
from datetime import datetime
from pathlib import Path
from types import TracebackType

HEADER = ("timestamp", "frame", "class", "count")


class EventLogger:
    """Append a row whenever a class count changes.

    Logging only the changes keeps the file readable: one row per frame would add
    thousands of identical rows every few minutes.
    """

    def __init__(self, path: Path, class_names: list[str]) -> None:
        self.path = path
        self._counts = {name: 0 for name in class_names}
        self._file = None
        self._writer = None

    def __enter__(self) -> "EventLogger":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        is_new_file = not self.path.exists() or self.path.stat().st_size == 0
        self._file = self.path.open("a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        if is_new_file:
            self._writer.writerow(HEADER)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._file.close()

    def update(self, frame_index: int, counts: dict[str, int]) -> None:
        """Write one row for each class whose count differs from the previous frame."""
        changes = [name for name, count in counts.items() if count != self._counts[name]]
        if not changes:
            return

        timestamp = datetime.now().isoformat(timespec="milliseconds")
        for name in changes:
            self._writer.writerow((timestamp, frame_index, name, counts[name]))
            self._counts[name] = counts[name]
        # Flushing means the events survive if the app is closed abruptly.
        self._file.flush()
