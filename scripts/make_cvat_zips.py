"""Package the raw Kaggle dataset into the two zips CVAT needs.

Creates:
    images.zip           the images, used when creating the CVAT task
    voc_annotations.zip  Pascal VOC boxes, imported into that task afterwards

Usage:
    python scripts/make_cvat_zips.py
"""

import argparse
import zipfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Read the input and output folders from the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/cvat_import"))
    return parser.parse_args()


def find_unmatched(images: list[Path], annotations: list[Path]) -> set[str]:
    """Return file stems that have an image or an annotation, but not both."""
    return {p.stem for p in images} ^ {p.stem for p in annotations}


def write_images_zip(images: list[Path], output_path: Path) -> None:
    """Zip images flat, so CVAT frame names equal the plain file names."""
    # PNGs are already compressed, so storing them is faster and no bigger.
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_STORED) as archive:
        for image in images:
            archive.write(image, arcname=image.name)


def write_voc_zip(annotations: list[Path], output_path: Path) -> None:
    """Zip annotations in the Pascal VOC layout that CVAT's importer expects."""
    image_list = "\n".join(p.stem for p in annotations) + "\n"
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for annotation in annotations:
            archive.write(annotation, arcname=f"Annotations/{annotation.name}")
        archive.writestr("ImageSets/Main/default.txt", image_list)


def main() -> None:
    """Check that images and annotations pair up, then write both zips."""
    args = parse_args()
    images = sorted((args.raw_dir / "images").glob("*.png"))
    annotations = sorted((args.raw_dir / "annotations").glob("*.xml"))

    if not images:
        raise SystemExit(f"No PNG images found in {args.raw_dir / 'images'}")
    unmatched = find_unmatched(images, annotations)
    if unmatched:
        raise SystemExit(f"{len(unmatched)} files have no pair, e.g. {sorted(unmatched)[:5]}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_images_zip(images, args.output_dir / "images.zip")
    write_voc_zip(annotations, args.output_dir / "voc_annotations.zip")
    print(f"Packaged {len(images)} images and {len(annotations)} annotations in {args.output_dir}")


if __name__ == "__main__":
    main()
