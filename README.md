# Outfit Compatibility Analysis

This repository contains the code, data-processing scripts, and experiment outputs for our end-to-end outfit compatibility system. The project combines YOLOv8-Seg garment segmentation, type-aware pairwise compatibility modeling, and outfit-level scoring, using the DeepFashion2 dataset.

Our full project report is included in **`OutfitOpsReport.pdf`**.

---

## Dataset Files

### DeepFashion2 Subsets
- **`subset_500.zip`**  
  https://drive.google.com/file/d/13oboCvpbsQ0Aw7JBMhTzgEcJ-hbPtrlW/view?usp=drive_link

- **`subset_5000_v2.zip`**  
  https://drive.google.com/file/d/1kDe3lI8o68XBEKO8a_LtF2DbNSFFpqId/view?usp=drive_link

- **`df2_subset_5000.zip`**  
  https://drive.google.com/file/d/1ZOJLFNtjlTdgfOpSvnB043iHU1fKBAw5/view?usp=drive_link

### Polyvore Subset
- **`polyvore_subset_5000.zip`**  
  https://drive.google.com/file/d/1-rxVqZt-77M7m1NxcYcMvl3zHJ4ovdia/view?usp=drive_link

---

## Model Output (too Large Files)

**Folder:** `resnet18-shop-weights`  
https://drive.google.com/drive/folders/1AhjRQxFGxLg8WHOshQA6cIKcZ1SiUJtX?usp=drive_link

---

## How to Run

The full pipeline—including dataset setup, YOLO segmentation, garment crop extraction, pair construction, compatibility model training, and outfit scoring—is documented inside:

**`Outfit_Compatibility_Code.ipynb`**

Open the notebook and follow the instructions inside.

---
