"""Convert a video into a small GIF for the README.

Usage:
    python scripts/video_to_gif.py outputs/demo.mp4 docs/images/demo.gif
"""

import argparse
from pathlib import Path

import cv2
from PIL import Image


def parse_args() -> argparse.Namespace:
    """Read the input, output and size settings from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=640, help="GIF width in pixels")
    parser.add_argument("--keep-every", type=int, default=3, help="keep one frame out of this many")
    return parser.parse_args()


def read_frames(path: Path, width: int, keep_every: int) -> tuple[list[Image.Image], float]:
    """Return the kept frames as RGB images, and the video's frame rate."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)

    frames = []
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % keep_every == 0:
            height = round(frame.shape[0] * width / frame.shape[1])
            small = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB)))
        index += 1
    capture.release()
    return frames, fps


def main() -> None:
    """Write the GIF, keeping the video's original playback speed."""
    args = parse_args()
    frames, fps = read_frames(args.input, args.width, args.keep_every)
    if not frames:
        raise SystemExit(f"No frames found in {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000 * args.keep_every / fps),
        loop=0,
        optimize=True,
    )
    size_mb = args.output.stat().st_size / 1_000_000
    print(f"Wrote {args.output}: {len(frames)} frames, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
