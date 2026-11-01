"""The command-line interface."""

import argparse
from dataclasses import replace
from pathlib import Path

from zone_watch.app import RunOptions, run
from zone_watch.config import AppConfig, load_config
from zone_watch.roi import Roi


def parse_source(value: str) -> int | str:
    """Turn a numeric source into a webcam index and leave file paths unchanged."""
    return int(value) if value.isdigit() else value


def parse_roi(value: str) -> Roi:
    """Parse a zone written as x1,y1,x2,y2."""
    try:
        x1, y1, x2, y2 = (int(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected x1,y1,x2,y2 but got '{value}'") from error
    return Roi.from_corners((x1, y1), (x2, y2))


def build_parser() -> argparse.ArgumentParser:
    """Define the command-line options."""
    parser = argparse.ArgumentParser(prog="zone_watch", description="Helmet detection inside a zone.")
    parser.add_argument("--source", type=parse_source, default=0, help="webcam index or video file path")
    parser.add_argument("--config", type=Path, default=Path("configs/app.yaml"))
    parser.add_argument("--model", type=Path, help="override model_path from the config")
    parser.add_argument("--confidence", type=float, help="override confidence from the config")
    parser.add_argument("--log", type=Path, help="override log_path from the config")
    parser.add_argument("--save-video", type=Path, help="write the annotated video to this file")
    parser.add_argument("--roi", type=parse_roi, help="start with this zone, as x1,y1,x2,y2")
    parser.add_argument("--no-display", action="store_true", help="run without a window; requires --roi")
    return parser


def apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Return the config with any values given on the command line replaced."""
    overrides = {
        "model_path": args.model,
        "confidence": args.confidence,
        "log_path": args.log,
    }
    return replace(config, **{key: value for key, value in overrides.items() if value is not None})


def main() -> None:
    """Parse the options and start the app."""
    parser = build_parser()
    args = parser.parse_args()
    if args.no_display and args.roi is None:
        parser.error("--no-display requires --roi, because there is no window to draw a zone in")

    config = apply_overrides(load_config(args.config), args)
    if not 0 <= config.confidence <= 1:
        parser.error(f"confidence must be between 0 and 1, got {config.confidence}")

    try:
        run(config, RunOptions(args.source, args.roi, not args.no_display, args.save_video))
    except KeyboardInterrupt:
        # Without a window, Ctrl+C is the normal way to stop a webcam run.
        print("Stopped.")
