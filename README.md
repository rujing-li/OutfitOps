# DeepFashion2 Outfit Parsing & Styling Pipeline

This project provides an **end-to-end system** for parsing outfit images, extracting garment-level features, generating outfit-level descriptors, and optionally producing **LLM-based styling suggestions**. It is built around a **fine-tuned YOLOv8s-seg model** trained on a curated subset of **DeepFashion2**. TODO: add outfit_compatibility/

---

## 1. Project Structure

```
├── deepfashion2_to_coco.py
│   Converts original DeepFashion2 annotations → COCO-style JSON.
│
├── make_subset.py
│   Creates a small subset (e.g., 500 images) for quicker training.
│
├── YOLO_finetuned_basic.ipynb
│   Fine-tunes YOLOv8s-seg on the subset.
│   Generates garment crops and color features.
│
├── YOLO_finetuned_llm.ipynb
│   Aggregates garment and color data → outfit-level JSON.
│   Provides inline visualizations.
│   Integrates LLM styling suggestions.
│
├── outfit_compatibility/   ← TODO
│   (Planned) outfit-compatibility scoring tools.
│
├── journal.md
│   Development notes and experiment logs.
│
└── data/
    DeepFashion2 directory (train/val/test)
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
