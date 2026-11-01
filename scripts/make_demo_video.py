"""Build a slideshow video from test images, as input for the app's demo.

The images are hand-picked to show the alert and the zone filter clearly, so the
demo illustrates behaviour. The test metrics in the README measure accuracy.

Usage:
    python scripts/make_demo_video.py
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 1280, 720
FPS = 25
SECONDS_PER_IMAGE = 1.5

DEMO_IMAGES = (
    "BikesHelmets273",
    "BikesHelmets404",
    "BikesHelmets99",
    "BikesHelmets25",
    "BikesHelmets526",
    "BikesHelmets378",
    "BikesHelmets379",
    "BikesHelmets642",
)


def parse_args() -> argparse.Namespace:
    """Read the input folder and output file from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images-dir", type=Path, default=Path("data/processed/images/test"))
    parser.add_argument("--output", type=Path, default=Path("videos/demo_source.mp4"))
    return parser.parse_args()


def fit_on_canvas(image: np.ndarray) -> np.ndarray:
    """Scale an image to fit the video frame and centre it on a black background."""
    height, width = image.shape[:2]
    scale = min(WIDTH / width, HEIGHT / height)
    resized = cv2.resize(image, (round(width * scale), round(height * scale)))
    canvas = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    top = (HEIGHT - resized.shape[0]) // 2
    left = (WIDTH - resized.shape[1]) // 2
    canvas[top:top + resized.shape[0], left:left + resized.shape[1]] = resized
    return canvas


def read_image(images_dir: Path, stem: str) -> np.ndarray:
    """Read one image by name, stopping with a clear message if it is missing."""
    path = images_dir / f"{stem}.png"
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"Could not read {path}")
    return image


def main() -> None:
    """Write each demo image to the video for a fixed time."""
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise SystemExit(f"Could not create {args.output}")

    for stem in DEMO_IMAGES:
        canvas = fit_on_canvas(read_image(args.images_dir, stem))
        for _ in range(round(FPS * SECONDS_PER_IMAGE)):
            writer.write(canvas)
    writer.release()
    print(f"Wrote {args.output} from {len(DEMO_IMAGES)} images")


if __name__ == "__main__":
    main()
