'''
How to Run:
1. Download polyvore_outfit dataset from: https://www.kaggle.com/datasets/enisteper1/polyvore-outfit-dataset
2. Extract zip file.
3. Replace ROOT_DIR with folder containing the folders 'disjoint', 'images', 'nondisjoint', 'categories.csv', 'polyvore_item_metadata.json', and 'polyvore_outfit_titles.json'.
4. Set N_OUTFITS
5. Run file to get Polyvore_subset_{N_OUTFITS} 
'''

import json
import csv
import random
import os
import shutil
from pathlib import Path

# Config
# path to polyvore_outfits folder
ROOT_DIR = Path(r"C:\Users\eliza\OneDrive\Documents\School\Columbia\Deep Learning for Computer Vision\Final Project\polyvore_outfits\polyvore_outfits")
N_OUTFITS = 5000
OUT_DIR_NAME = f"polyvore_subset_{N_OUTFITS}"

OUTFIT_JSONS = [
    ROOT_DIR / "disjoint" / "train.json",
    ROOT_DIR / "disjoint" / "valid.json",
    ROOT_DIR / "disjoint" / "test.json",
]

ITEM_META_PATH = ROOT_DIR / "polyvore_item_metadata.json"
CATEGORIES_CSV = ROOT_DIR / "categories.csv"
IMAGES_ROOT = ROOT_DIR / "images"

# COARSE TYPE MAPPING
COARSE_TYPES = ["top", "bottom", "outerwear", "dress"]
TYPE2ID = {t: i for i, t in enumerate(COARSE_TYPES)}

POLYVORE_MAIN_TO_COARSE = {
    "tops": "top",
    "bottoms": "bottom",
    "outerwear": "outerwear",
    "all-body": "dress",
}

def build_category_lookup(categories_csv_path: Path):
    """category_id (str) -> {'sub_category': ..., 'main_category': ...}"""
    lookup = {}
    with categories_csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row["category_id"]
            lookup[cid] = {
                "sub_category": row["sub_category"],
                "main_category": row["main_category"],
            }
    return lookup

# Convert to Coarse Types for each item ('top'/'bottom'/'outerwear'/'dress' or None)
def meta_to_coarse_type(meta, cat_lookup):
    cat_id = str(meta.get("category_id"))
    if cat_id in cat_lookup:
        main_cat = cat_lookup[cat_id]["main_category"]
        if main_cat in POLYVORE_MAIN_TO_COARSE:
            return POLYVORE_MAIN_TO_COARSE[main_cat]

    sc = meta.get("semantic_category")
    mc = meta.get("main_category")
    for cat in (sc, mc):
        if cat in POLYVORE_MAIN_TO_COARSE:
            return POLYVORE_MAIN_TO_COARSE[cat]

    sub = (meta.get("sub_category") or "").lower()
    title = (meta.get("title") or "").lower()
    desc = (meta.get("description") or "").lower()
    text = " ".join([sub, title, desc])

    if any(k in text for k in ["dress", "jumpsuit", "romper"]):
        return "dress"
    if any(k in text for k in ["skirt", "shorts", "pants", "jeans", "trouser"]):
        return "bottom"
    if any(k in text for k in ["coat", "jacket", "blazer", "cardigan", "outerwear"]):
        return "outerwear"
    if any(k in text for k in ["shirt", "top", "tee", "t-shirt", "blouse", "sweater", "hoodie"]):
        return "top"

    return None


def main():
    random.seed(42)

    out_dir = ROOT_DIR / OUT_DIR_NAME
    out_dir.mkdir(exist_ok=True)
    out_images_dir = out_dir / "images"
    out_images_dir.mkdir(exist_ok=True)

    print(f"Subset folder: {out_dir}")

    print("Loading item metadata…")
    with ITEM_META_PATH.open("r", encoding="utf-8") as f:
        item_meta_all = json.load(f)

    print("Loading categories.csv…")
    cat_lookup = build_category_lookup(CATEGORIES_CSV)

    item_type = {}
    item_img_src = {}

    print("Scanning items for valid type + image…")
    for item_id, meta in item_meta_all.items():
        coarse = meta_to_coarse_type(meta, cat_lookup)
        if coarse is None:
            continue

        candidate_paths = [
            IMAGES_ROOT / f"{item_id}.jpg",
            IMAGES_ROOT / f"{item_id}.png",
            IMAGES_ROOT / f"{item_id}.jpeg",
        ]
        img_path = None
        for p in candidate_paths:
            if p.exists():
                img_path = p
                break

        if img_path is None:
            continue

        item_type[item_id] = coarse
        item_img_src[item_id] = img_path

    print(f"Usable items (have coarse type + image file): {len(item_type):,}")

    all_outfits = []
    for path in OUTFIT_JSONS:
        print(f"Loading outfits from {path}…")
        with path.open("r", encoding="utf-8") as f:
            outfits = json.load(f)

        for outfit in outfits:
            item_ids = []
            if "items" in outfit and isinstance(outfit["items"], list):
                for it in outfit["items"]:
                    if isinstance(it, dict):
                        iid = it.get("item_id")
                    else:
                        iid = it
                    if iid in item_type:
                        item_ids.append(iid)
            else:
                item_ids = [iid for iid in outfit if iid in item_type]

            if len(item_ids) >= 2:
                all_outfits.append({
                    "set_id": outfit.get("set_id"),
                    "item_ids": item_ids,
                })

    print(f"Usable outfits (>=2 valid items): {len(all_outfits):,}")

    if len(all_outfits) < N_OUTFITS:
        print(f"WARNING: only {len(all_outfits)} outfits available; "
              f"using all instead of requested {N_OUTFITS}.")
        n_sub = len(all_outfits)
    else:
        n_sub = N_OUTFITS

    random.shuffle(all_outfits)
    subset_outfits = all_outfits[:n_sub]
    print(f"Sampled {len(subset_outfits):,} outfits for subset.")

    subset_item_ids = set()
    for o in subset_outfits:
        subset_item_ids.update(o["item_ids"])

    print(f"Unique items in subset outfits: {len(subset_item_ids):,}")

    subset_items = {}
    copied = 0

    for iid in subset_item_ids:
        src = item_img_src[iid]
        dst = out_images_dir / src.name

        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1

        meta = item_meta_all[iid].copy()
        meta["coarse_type"] = item_type[iid]
        meta["image_relpath"] = str(Path("images") / dst.name)
        subset_items[iid] = meta

    print(f"Copied {copied:,} image files into {out_images_dir}")

    # 90/10 train/val split over outfits
    split_idx = int(0.9 * len(subset_outfits))
    train_outfits = subset_outfits[:split_idx]
    val_outfits = subset_outfits[split_idx:]

    print(f"Train outfits: {len(train_outfits):,}")
    print(f"Val outfits:   {len(val_outfits):,}")

    # subset JSONs
    items_out_path = out_dir / "items_subset.json"
    train_out_path = out_dir / "outfits_train_subset.json"
    val_out_path = out_dir / "outfits_val_subset.json"

    with items_out_path.open("w", encoding="utf-8") as f:
        json.dump(subset_items, f, indent=2)

    with train_out_path.open("w", encoding="utf-8") as f:
        json.dump(train_outfits, f, indent=2)

    with val_out_path.open("w", encoding="utf-8") as f:
        json.dump(val_outfits, f, indent=2)

    print("Finished.")
    print("Item metadata subset:", items_out_path)
    print("Train outfits subset:", train_out_path)
    print("Val outfits subset:  ", val_out_path)
    print("Images copied to:    ", out_images_dir)


if __name__ == "__main__":
    main()
