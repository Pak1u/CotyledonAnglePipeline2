from ultralytics import YOLO

def main():
    # Load the base pose model
    model = YOLO('yolov8n-pose.pt')

    # Start optimized training
    model.train(
        data='data.yaml',
        epochs=150,        # Increased epochs to allow for better convergence
        imgsz=640,         # Standard resolution for high detail
        batch=16,          # Increased batch size for more stable gradients
        workers=4,          # Leveraging multi-threading since system is stable
        device=0,          # Using your 1660 Super
        project='SaplingProject',
        name='v1_final_clean',
        
        # --- Advanced Hyperparameters ---
        optimizer='AdamW', # Often better for pose estimation on small datasets
        lr0=0.01,          # Initial learning rate
        patience=50,       # Stops early if no improvement for 50 epochs
        amp=False,         # Keeping False to avoid the NaN loss errors seen earlier
        
        # --- Augmentation (Crucial for small datasets) ---
        mosaic=1.0,        # Combines images to help the model see different scales
        mixup=0.1,         # Blends images to reduce overfitting
        flipud=0.5,        # Vertical flip for different growth angles
        fliplr=0.5,        # Horizontal flip
        degrees=15.0,      # Slight rotations to simulate natural variation
        
        # --- Loss Weights ---
        pose=12.0,         # Higher weight on keypoint accuracy
        kobj=1.0           # Objectness weight for the sapling itself
    )

if __name__ == '__main__':
    main()