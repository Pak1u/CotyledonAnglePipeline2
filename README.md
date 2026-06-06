# Cotyledon Angle Pipeline

This project measures plant cotyledon angles from images using a YOLOv8 pose model. The trained model predicts five keypoints per sapling, then the code calculates two botanical measurements:

- Stalk angle: left cotyledon base -> central junction -> right cotyledon base
- Tip angle: left cotyledon tip -> central junction -> right cotyledon tip

## Keypoint Order

The YOLO labels use this fixed order:

1. Left cotyledon tip
2. Left cotyledon base
3. Right cotyledon base
4. Right cotyledon tip
5. Central junction

The shared helpers in `cotyledon_angle/` name these points before measuring angles, which keeps the math consistent across training checks, inference, and CSV export.

## Main Files

- `train.py`: fine-tunes `yolov8n-pose.pt` on the sapling pose dataset.
- `inference.py`: displays random validation predictions with keypoint and angle overlays.
- `export_results.py`: runs the trained model over train/val images and writes `sapling_cleaned_data.csv`.
- `data.yaml`: YOLO dataset configuration.
- `cotyledon_angle/geometry.py`: shared keypoint mapping and angle calculations.
- `cotyledon_angle/visualization.py`: shared OpenCV overlay drawing.
- `cotyledon_angle/paths.py`: repository-relative default paths.

## Legacy Utilities

- `test_val.py`: visual checker for the older `v1_manual_points-4` weights.
- `ground_truth.py`: overlays recovered manual annotations for spot-checking.
- `sam_slicer.py`: OpenCV crop extractor for older full-image workflows.
- `measurer.py`: optional Gemini-based crop measurer kept as an experimental fallback.

## Typical Workflow

Train:

```bash
python3 train.py
```

Inspect validation predictions:

```bash
python3 inference.py --samples 10 --conf 0.6
```

Export measured angles:

```bash
python3 export_results.py --output sapling_cleaned_data.csv
```

## Notes

The repository keeps useful trained weights and result summaries under `runs/pose/SaplingProject/`. Temporary YOLO cache files are ignored because they are regenerated automatically.
