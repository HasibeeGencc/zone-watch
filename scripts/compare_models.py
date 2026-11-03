"""Compare trained models on accuracy and CPU speed, and print a Markdown table.

Every model is measured on this computer, with the same images and settings.

Choose between models using the val row: choosing with the test row would make the
test scores optimistic. Report the test row for the chosen model.

Usage:
    python scripts/compare_models.py models/yolo11n.pt models/yolo11s.pt
"""

import argparse
import statistics
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

IMAGE_SIZE = 640
WARMUP_RUNS = 5
FRAME_SIZE = (1280, 720)


def parse_args() -> argparse.Namespace:
    """Read the models and the dataset from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", type=Path, nargs="+", help="weights files to compare")
    parser.add_argument("--data", type=Path, default=Path("data/processed/data.yaml"))
    parser.add_argument("--class-name", default="Without Helmet", help="class to report separately")
    return parser.parse_args()


def measure_accuracy(model: YOLO, data: Path, split: str, class_name: str) -> dict[str, float]:
    """Return mAP for all classes, and precision and recall for one class, on a split."""
    metrics = model.val(
        data=str(data.resolve()), split=split, imgsz=IMAGE_SIZE, device="cpu",
        workers=0, plots=False, verbose=False,
    ).box
    class_ids = {model.names[int(class_id)]: index for index, class_id in enumerate(metrics.ap_class_index)}
    index = class_ids[class_name]
    return {
        "map50": metrics.map50,
        "map": metrics.map,
        "class_precision": metrics.p[index],
        "class_recall": metrics.r[index],
        "class_map50": metrics.ap50[index],
    }


def measure_inference_ms(model: YOLO, images: list[Path]) -> float:
    """Return the median time for one prediction on a video-sized frame, in milliseconds."""
    frames = [cv2.resize(cv2.imread(str(image)), FRAME_SIZE) for image in images]
    for _ in range(WARMUP_RUNS):
        model.predict(frames[0], imgsz=IMAGE_SIZE, device="cpu", verbose=False)

    times = []
    for frame in frames:
        start = time.perf_counter()
        model.predict(frame, imgsz=IMAGE_SIZE, device="cpu", verbose=False)
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)


def main() -> None:
    """Measure each model and print one table row per model and split."""
    args = parse_args()
    missing = [path for path in args.models if not path.exists()]
    if missing:
        raise SystemExit(f"Model files not found: {missing}")
    images = sorted((args.data.parent / "images" / "test").iterdir())

    rows = []
    for path in args.models:
        model = YOLO(str(path))
        inference_ms = measure_inference_ms(model, images)
        size_mb = path.stat().st_size / 1_000_000
        for split in ("val", "test"):
            scores = measure_accuracy(model, args.data, split, args.class_name)
            rows.append(
                f"| {path.stem} | {split} | {scores['map50']:.3f} | {scores['map']:.3f} "
                f"| {scores['class_precision']:.3f} | {scores['class_recall']:.3f} "
                f"| {scores['class_map50']:.3f} | {1000 / inference_ms:.1f} | {size_mb:.1f} |"
            )

    print(f"| Model | Split | mAP@50 | mAP@50-95 | {args.class_name} P | {args.class_name} R "
          f"| {args.class_name} mAP@50 | FPS (CPU, model only) | Size (MB) |")
    print("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
