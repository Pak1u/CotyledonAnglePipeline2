"""Train the YOLOv8 pose model used for cotyledon angle prediction."""

from ultralytics import YOLO

from cotyledon_angle.paths import BASE_POSE_MODEL, DATA_YAML, PROJECT_ROOT

def main():
    # Start from YOLOv8n-pose and fine-tune it on the five sapling landmarks
    # described in data.yaml.
    model = YOLO(str(BASE_POSE_MODEL))

    # These settings are tuned for a small curated sapling dataset. AMP is off
    # because earlier runs hit NaN losses with mixed precision.
    model.train(
        data=str(DATA_YAML),
        epochs=150,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        project=str(PROJECT_ROOT / 'runs' / 'pose' / 'SaplingProject'),
        name='v1_final_clean',
        
        # AdamW tends to behave well for small pose datasets with sparse labels.
        optimizer='AdamW',
        lr0=0.01,
        patience=50,
        amp=False,
        
        # Augmentation helps the model generalize to plant scale, orientation,
        # and lighting variation seen across trays/images.
        mosaic=1.0,
        mixup=0.1,
        flipud=0.5,
        fliplr=0.5,
        degrees=15.0,
        
        # Keypoint accuracy matters more than the bounding box for angle output.
        pose=12.0,
        kobj=1.0
    )

if __name__ == '__main__':
    main()
