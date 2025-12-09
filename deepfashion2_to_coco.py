import json
import os
from pathlib import Path
from typing import Dict, List, Any

from PIL import Image
from tqdm import tqdm


# ---------- Category mapping ----------

# DeepFashion2: category_id 1..13
DF2_CATEGORIES = {
    1: "short_sleeve_top",
    2: "long_sleeve_top",
    3: "short_sleeve_outwear",
    4: "long_sleeve_outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short_sleeve_dress",
    11: "long_sleeve_dress",
    12: "vest_dress",
    13: "sling_dress",
}


def build_categories() -> List[Dict[str, Any]]:
    """Build COCO-style categories list."""
    categories = []
    for cid, name in DF2_CATEGORIES.items():
        categories.append({
            "id": cid,
            "name": name,
            "supercategory": "clothes",
        })
    return categories


# ---------- Core conversion ----------

def convert_split_to_coco(split_root: Path, out_json: Path) -> None:
    """
    Convert a single DeepFashion2 split (train/ or val/ or test/) to COCO.

    split_root: path to folder containing "image/" and "annos/".
    out_json:   path to write the COCO JSON file.
    """
    annos_dir = split_root / "annos"
    images_dir = split_root / "image"

    if not annos_dir.is_dir():
        raise FileNotFoundError(f"Annotations dir not found: {annos_dir}")
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images dir not found: {images_dir}")

    image_entries = []
    annotation_entries = []
    categories = build_categories()

    img_id = 1
    ann_id = 1

    json_files = sorted([p for p in annos_dir.iterdir() if p.suffix == ".json"])
    print(f"[{split_root.name}] Found {len(json_files)} annotation files.")

    for jf in tqdm(json_files, desc=f"Converting {split_root.name}"):
        with jf.open("r") as f:
            data = json.load(f)

        stem = jf.stem  # e.g., "000001"
        img_name_jpg = stem + ".jpg"
        img_path = images_dir / img_name_jpg

        if not img_path.is_file():
            # Some datasets may have .png; you can add a fallback here if needed
            print(f"Warning: image file not found for {jf.name}, skipping.")
            continue

        # Get image size
        with Image.open(img_path) as im:
            width, height = im.size

        # COCO image entry
        image_entries.append({
            "id": img_id,
            "file_name": img_name_jpg,
            "width": width,
            "height": height,
        })

        # Each "itemK" is a clothing item
        for key, item in data.items():
            if not key.startswith("item"):
                continue

            # Required fields
            cat_id = int(item["category_id"])
            if cat_id not in DF2_CATEGORIES:
                # Skip unknown category
                continue

            bbox = item["bounding_box"]  # [x1, y1, x2, y2]
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = bbox
            w = max(0.0, float(x2) - float(x1))
            h = max(0.0, float(y2) - float(y1))
            if w <= 0 or h <= 0:
                continue

            coco_bbox = [float(x1), float(y1), w, h]

            seg = item.get("segmentation", [])
            # seg can be:
            #   [x1,y1,...]  -> single polygon
            #   [[x1,y1,...], [ ... ]] -> multiple polygons
            #   [] or missing
            coco_seg = []

            if isinstance(seg, list) and len(seg) > 0:
                # If first element is a number, wrap as single polygon
                if isinstance(seg[0], (int, float)):
                    seg = [seg]

                for poly in seg:
                    # Ensure even length and at least 3 points (6 numbers)
                    if not isinstance(poly, list):
                        continue
                    if len(poly) < 6 or len(poly) % 2 != 0:
                        continue
                    coco_seg.append([float(v) for v in poly])

            # If you want to keep items without segmentation, you can:
            # - either approximate seg with bbox,
            # - or skip them. Here we **skip** items with no valid polygons.
            if len(coco_seg) == 0:
                # Uncomment this line if you prefer bbox-only training:
                # coco_seg = [[x1, y1, x2, y1, x2, y2, x1, y2]]
                # For now, we skip them:
                continue

            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": coco_bbox,
                "segmentation": coco_seg,
                "area": w * h,
                "iscrowd": 0,
            }
            annotation_entries.append(ann)
            ann_id += 1

        img_id += 1

    coco = {
        "images": image_entries,
        "annotations": annotation_entries,
        "categories": categories,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w") as f:
        json.dump(coco, f)
    print(f"Saved COCO annotations for {split_root.name} to: {out_json}")
    print(f"  images:      {len(image_entries)}")
    print(f"  annotations: {len(annotation_entries)}")
    print(f"  categories:  {len(categories)}")


def convert_deepfashion2(root: str, splits=("train", "val", "test")) -> None:
    """
    Convert multiple splits under a DeepFashion2 root folder.

    root: path to DeepFashion2 (containing train/, val/, test/).
    splits: which subfolders to convert.
    """
    root_path = Path(root)

    for split in splits:
        split_root = root_path / split
        if not split_root.is_dir():
            print(f"Skip split '{split}' (folder not found: {split_root})")
            continue

        out_json = root_path / f"{split}_coco.json"
        convert_split_to_coco(split_root, out_json)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert DeepFashion2 (train/val/test) to COCO JSON."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to DeepFashion2 root folder (containing train/, val/, test/).",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val", "test"],
        help="Which splits to convert (default: train val test).",
    )
    args = parser.parse_args()

    convert_deepfashion2(args.root, splits=args.splits)

# python deepfashion2_to_coco.py \
#   --root 'DeepFashion2' \
#   --splits train val

# python deepfashion2_to_coco.py --root "C:\Users\eliza\Downloads\train" --splits train
