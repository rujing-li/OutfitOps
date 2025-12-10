# OutfitOps

This project provides an end-to-end system for parsing outfit images, extracting garment-level features, producing LLM-based styling suggestions, and running an end-to-end **outfit compatibility** system. The project combines YOLOv8-Seg garment segmentation, type-aware pairwise compatibility modeling, and outfit-level scoring, using the DeepFashion2 dataset.

---

## 1. Project Structure

This repo contains the code, data-processing scripts, and experiment outputs.

```
├── Outfit_Compatibility_Outputs/
│   Reference outputs from compatibility scoring experiments.
│
├── Subset_Extraction_Code/
│   Scripts for generating DeepFashion2 subsets.
│
├── Outfit_Compatibility_Code.ipynb
│   Notebook for computing outfit compatibility scores.
│   Uses garment-level + outfit-level features.
│
├── YOLO_finetuned_basic.ipynb
│   Fine-tunes YOLOv8s-seg on DeepFashion2 subset.
│   Generates garment crops and color features.
│
├── YOLO_finetuned_llm.ipynb
    Aggregates garment metadata into outfit-level JSON.
    Integrates LLM styling suggestions.
    Provides inline visualizations.
```

---

## 2. Workflow 

### Step 1 — Dataset

Place the DeepFashion2 dataset here:

```
DeepFashion2/
  ├── train/
  ├── val/
  ├── test/
```
### Step 2 — Convert Annotations and Create Training Subset

Run `deepfashion2_to_coco.py` to generate COCO annotations compatible with YOLO training.
Use `make_subset.py` to extract a manageable subset for fast experimentation.

### Step 3 — Run each ipynb files seperately

Note the data preprocessing scripts can run locally where DeepFasion2 exist. However, the two Jupyter Notebooks are tested in Google Colab and assume the dataset + JSONs are already prepared.

Note: The full pipeline—including dataset setup, YOLO segmentation, garment crop extraction, pair construction, compatibility model training, and outfit scoring—is documented inside: **`Outfit_Compatibility_Code.ipynb`** Open the notebook and follow the instructions inside.


## 3. Links to Files

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


### Model Output
**Folder:** `resnet18-shop-weights`  
https://drive.google.com/drive/folders/1AhjRQxFGxLg8WHOshQA6cIKcZ1SiUJtX?usp=drive_link

---
