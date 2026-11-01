"""Load the app settings from a YAML file."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class AppConfig:
    """Settings that stay the same for a whole run."""

    model_path: Path
    confidence: float
    image_size: int
    device: str | None
    alert_class: str
    log_path: Path


def load_config(path: Path) -> AppConfig:
    """Read and validate the YAML config file, stopping with a clear message if it is wrong."""
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")

    with path.open(encoding="utf-8") as file:
        values = yaml.safe_load(file)

    try:
        config = AppConfig(**values)
    except TypeError as error:
        raise SystemExit(f"Invalid config file {path}: {error}") from error

    # YAML reads these as plain strings, and a device like 0 as a number.
    return AppConfig(
        model_path=Path(config.model_path),
        confidence=config.confidence,
        image_size=config.image_size,
        device=None if config.device is None else str(config.device),
        alert_class=config.alert_class,
        log_path=Path(config.log_path),
    )
