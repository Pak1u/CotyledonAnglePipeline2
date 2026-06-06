"""Show random validation predictions with cotyledon angle overlays."""

import argparse
import glob
import os
import random
from pathlib import Path

import cv2
from ultralytics import YOLO

from cotyledon_angle.geometry import CotyledonPoints, cotyledon_angles
from cotyledon_angle.paths import DATASET_DIR, FINAL_MODEL
from cotyledon_angle.visualization import draw_yolo_measurement


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=FINAL_MODEL, help="YOLOv8 pose weights to use.")
    parser.add_argument(
        "--val-dir",
        type=Path,
        default=DATASET_DIR / "images" / "val",
        help="Validation image directory.",
    )
    parser.add_argument("--samples", type=int, default=10, help="Number of random validation images to display.")
    parser.add_argument("--conf", type=float, default=0.6, help="Prediction confidence threshold.")
    return parser.parse_args()


def run_clean_inference(model_path: Path, val_dir: Path, samples: int, conf: float):
    """Run model predictions on a small random validation sample."""
    model = YOLO(str(model_path))

    # This expects the held-out validation images from the 80/20 YOLO split.
    all_val_imgs = glob.glob(os.path.join(str(val_dir), "*.jpg"))
    
    if not all_val_imgs:
        print(f"Error: Could not find images in {val_dir}")
        return

    sample_imgs = random.sample(all_val_imgs, min(samples, len(all_val_imgs)))
    print(f"--- Running Inference on {len(sample_imgs)} Cleaned Validation Images ---")
    print("Press any key to jump to the next image. Press 'q' to quit early.")

    for img_path in sample_imgs:
        results = model(img_path, conf=conf, verbose=False)
        
        for r in results:
            img = r.orig_img.copy()
            
            if r.keypoints is None or len(r.keypoints.xy[0]) < 5:
                print(f"Skipping {os.path.basename(img_path)}: 5 points not detected.")
                continue

            # Convert raw YOLO keypoints into named botanical landmarks before
            # measuring; this prevents index mistakes when the label order changes.
            pts = r.keypoints.xy[0].cpu().numpy()
            cotyledon_points = CotyledonPoints.from_yolo(pts)
            stalk_ang, tip_ang = cotyledon_angles(cotyledon_points)
            draw_yolo_measurement(img, cotyledon_points, stalk_ang, tip_ang)

            cv2.imshow("Fixed YOLO Prediction", img)
            print(f"Showing: {os.path.basename(img_path)}")
            
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return
                
    cv2.destroyAllWindows()
    print("Done showing the 10 showcase images!")

if __name__ == "__main__":
    args = parse_args()
    run_clean_inference(args.model, args.val_dir, args.samples, args.conf)
