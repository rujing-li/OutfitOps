#!/usr/bin/env python3
import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, Any, List

from tqdm import tqdm


def load_coco(coco_path: Path) -> Dict[str, Any]:
    with coco_path.open("r") as f:
        return json.load(f)


def sample_images(
    images: List[Dict[str, Any]],
    n_images: int,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    n = min(n_images, len(images))
    return random.sample(images, n)


def filter_annotations(
    annotations: List[Dict[str, Any]],
    image_ids: set,
) -> List[Dict[str, Any]]:
    return [ann for ann in annotations if ann["image_id"] in image_ids]


def copy_images(
    sampled_images: List[Dict[str, Any]],
    src_img_dir: Path,
    dst_img_dir: Path,
) -> None:
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    for img in tqdm(sampled_images, desc="Copying images"):
        src = src_img_dir / img["file_name"]
        dst = dst_img_dir / img["file_name"]
        if not src.is_file():
            print(f"Warning: image file not found: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_subset_coco(
    sampled_images: List[Dict[str, Any]],
    sampled_annotations: List[Dict[str, Any]],
    categories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "images": sampled_images,
        "annotations": sampled_annotations,
        "categories": categories,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a smaller COCO-style subset from a larger dataset."
    )

    parser.add_argument(
        "--coco-json",
        type=str,
        required=True,
        help="Path to the original COCO JSON file (e.g., train_coco.json).",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Path to the directory containing the images referenced in the COCO JSON.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Output directory for the subset (will contain images/ and annotations.json).",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=500,
        help="Number of images to sample (default: 500).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42).",
    )

    args = parser.parse_args()

    coco_path = Path(args.coco_json)
    src_img_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_img_dir = out_dir / "images"
    out_ann_path = out_dir / "annotations.json"

    if not coco_path.is_file():
        raise FileNotFoundError(f"COCO JSON not found: {coco_path}")
    if not src_img_dir.is_dir():
        raise NotADirectoryError(f"Images directory not found: {src_img_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading COCO from: {coco_path}")
    coco = load_coco(coco_path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    print(f"Total images in dataset: {len(images)}")
    print(f"Total annotations in dataset: {len(annotations)}")

    print(f"Sampling up to {args.n_images} images with seed={args.seed}...")
    sampled_images = sample_images(images, args.n_images, seed=args.seed)
    sampled_ids = {img["id"] for img in sampled_images}

    print("Filtering annotations for sampled images...")
    sampled_annotations = filter_annotations(annotations, sampled_ids)

    print(f"Sampled images: {len(sampled_images)}")
    print(f"Sampled annotations: {len(sampled_annotations)}")

    print("Copying image files...")
    copy_images(sampled_images, src_img_dir, out_img_dir)

    print("Writing subset COCO JSON...")
    subset_coco = build_subset_coco(sampled_images, sampled_annotations, categories)
    with out_ann_path.open("w") as f:
        json.dump(subset_coco, f)

    print("Done.")
    print(f"Subset written to: {out_dir}")
    print(f"  - images: {out_img_dir}")
    print(f"  - annotations: {out_ann_path}")


if __name__ == "__main__":
    main()


# python make_subset.py \
#   --coco-json DeepFashion2/train_coco.json \
#   --images-dir DeepFashion2/train/image \
#   --out-dir subset_5000_v2 \
#   --n-images 5000 \
#   --seed 42
