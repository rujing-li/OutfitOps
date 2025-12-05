# Project Progress

## Completed Steps
- **DeepFashion2 Data Password Issues** — Solved
- **GCP Server Internet Access** — Solved

## Blocked/In Progress
- **Google Colab Data Open and Extract** — Blocked
- **Macbook Pro MPS and Detectron2** — Blocked
- **GCP Server Space/Disk** — Temporarily blocked (tried)
- **Google Colab with Small Subset + Detectron2** — Blocked

## Abandoned Approaches
- **Macbook Pro MPS and YOLO** — Abandoned

## Not Yet Attempted
- **Vast.ai Server** — Not tried

## Successful Approach
- **Google Colab with Small Subset + YOLO** — Successful

---

# Next Steps

## Model Fine-Tuning
### Current Approach
We fine-tuned all parameters of YOLOv8s-seg on our DeepFashion2 subset without freezing any layers. This allows the model to fully adapt its feature extractor and segmentation heads to fashion-specific garments rather than generic COCO categories. This approach is justified because COCO's garment categories (e.g., "person," "handbag") differ substantially from DeepFashion2's fine-grained taxonomy (13 clothing classes). Full-model updates help it learn color, texture, and clothing-specific features.
### Options to Explore
- Experiment with partial layer freezing (might now work well on YOLO)
- Adjust learning rates for different model components
- Test different backbone architectures 
  - Detectron2 (Mask R-CNN or RetinaMask)
  - A 2-stage pipeline:
    - Stage 1: DeepLabV3 / U-Net for segmentation
    - Stage 2: ResNet-18 / EfficientNet-B0 for garment classification

## Data Expansion (!)
- Add more data from DeepFashion2 (current: 100 images)
- Explore alternative datasets


## Additional Features (!!)
- Train a style classifier using labeled datasets (with style labels)
- Expand class: add shoes, hats, accessories to current 12 classes

## Deployment (!)
- Develop demo code