"""Turn a CVAT Ultralytics YOLO export into a train/val/test dataset with a data.yaml.

Labels come from the CVAT export and images from the raw dataset, so the export
does not need its own copy of the images. The split is stratified and reproducible:
the same seed always gives the same files in each split.

Usage:
    python scripts/prepare_dataset.py
"""

import argparse
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import yaml

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")


def parse_args() -> argparse.Namespace:
    """Read paths, split ratios and the seed from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export-dir", type=Path, default=Path("data/cvat_export"))
    parser.add_argument("--images-dir", type=Path, default=Path("data/raw/images"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_class_names(export_dir: Path) -> dict[int, str]:
    """Return the class id to name mapping from the export's data.yaml."""
    with (export_dir / "data.yaml").open(encoding="utf-8") as file:
        return yaml.safe_load(file)["names"]


def read_image_stems(export_dir: Path) -> list[str]:
    """Return the name, without extension, of every image listed in the export."""
    lines = (export_dir / "train.txt").read_text(encoding="utf-8").splitlines()
    return sorted(PurePosixPath(line.strip()).stem for line in lines if line.strip())


def find_image(images_dir: Path, stem: str) -> Path:
    """Return the path of the image with this stem, or stop if it does not exist."""
    for suffix in IMAGE_SUFFIXES:
        path = images_dir / f"{stem}{suffix}"
        if path.exists():
            return path
    raise SystemExit(f"No image found for '{stem}' in {images_dir}")


def has_valid_coordinates(line: str, num_classes: int) -> bool:
    """Check a YOLO label line: stop if malformed, return whether coordinates are in 0-1."""
    fields = line.split()
    if len(fields) != 5 or not 0 <= int(fields[0]) < num_classes:
        raise SystemExit(f"Malformed label line: {line!r}")
    return all(0 <= float(value) <= 1 for value in fields[1:])


def read_labels(path: Path) -> list[str]:
    """Return an image's label lines.

    A missing file is valid: CVAT writes none for an image without boxes, and YOLO
    uses such images as background examples.
    """
    if not path.exists():
        return []
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stratified_split(
    labels_by_stem: dict[str, list[str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, list[str]]:
    """Split images so that each split gets a fair share of every class combination.

    Images are grouped by the classes they contain and each group is split separately.
    This stops the rarer "Without Helmet" class from being unevenly distributed by chance.
    """
    groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for stem in sorted(labels_by_stem):
        classes = {label.split()[0] for label in labels_by_stem[stem]}
        groups[tuple(sorted(classes))].append(stem)

    rng = random.Random(seed)
    split: dict[str, list[str]] = {name: [] for name in SPLITS}
    for key in sorted(groups):
        stems = groups[key]
        rng.shuffle(stems)
        train_end = round(len(stems) * train_ratio)
        val_end = train_end + round(len(stems) * val_ratio)
        split["train"] += stems[:train_end]
        split["val"] += stems[train_end:val_end]
        split["test"] += stems[val_end:]
    return split


def write_split(
    split: dict[str, list[str]],
    labels_by_stem: dict[str, list[str]],
    images_dir: Path,
    output_dir: Path,
) -> None:
    """Copy each image and write its label file into the YOLO folder layout."""
    for name, stems in split.items():
        image_dir = output_dir / "images" / name
        label_dir = output_dir / "labels" / name
        image_dir.mkdir(parents=True)
        label_dir.mkdir(parents=True)
        for stem in stems:
            image = find_image(images_dir, stem)
            shutil.copy2(image, image_dir / image.name)
            if labels_by_stem[stem]:
                content = "\n".join(labels_by_stem[stem]) + "\n"
                (label_dir / f"{stem}.txt").write_text(content, encoding="utf-8")


def write_data_yaml(output_dir: Path, class_names: dict[int, str]) -> None:
    """Write the file that tells Ultralytics where the data is and what the classes are.

    There is no 'path' key, so the paths stay relative to wherever this file ends up.
    """
    config = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": class_names,
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)


def print_summary(
    split: dict[str, list[str]],
    labels_by_stem: dict[str, list[str]],
    class_names: dict[int, str],
) -> None:
    """Print how many images and boxes of each class ended up in each split."""
    for name in SPLITS:
        boxes = Counter(
            int(label.split()[0]) for stem in split[name] for label in labels_by_stem[stem]
        )
        counts = ", ".join(f"{class_names[i]}: {boxes[i]}" for i in sorted(class_names))
        print(f"{name:>5}: {len(split[name])} images | {counts}")


def main() -> None:
    """Read and check the labels, split them, then write the dataset."""
    args = parse_args()
    if not 0 < args.train_ratio + args.val_ratio < 1:
        raise SystemExit("train-ratio + val-ratio must be between 0 and 1")
    # Refuse to mix a new split with old files, which would leak images between splits.
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit(f"{args.output_dir} is not empty. Delete it first to rebuild the split.")

    class_names = read_class_names(args.export_dir)
    labels_by_stem: dict[str, list[str]] = {}
    excluded: list[str] = []
    for stem in read_image_stems(args.export_dir):
        labels = read_labels(args.export_dir / "labels" / "train" / f"{stem}.txt")
        # Some Kaggle boxes have corrupt coordinates that cannot be recovered. Dropping
        # only the box would teach the model that the helmet in it is background, so
        # the whole image is left out until its boxes are fixed in CVAT.
        if all(has_valid_coordinates(label, len(class_names)) for label in labels):
            labels_by_stem[stem] = labels
        else:
            excluded.append(stem)

    split = stratified_split(labels_by_stem, args.train_ratio, args.val_ratio, args.seed)
    write_split(split, labels_by_stem, args.images_dir, args.output_dir)
    write_data_yaml(args.output_dir, class_names)

    print_summary(split, labels_by_stem, class_names)
    print(f"Excluded {len(excluded)} images with invalid coordinates: {excluded}")
    print(f"Dataset written to {args.output_dir}")


if __name__ == "__main__":
    main()
