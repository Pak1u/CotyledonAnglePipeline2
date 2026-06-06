"""Export cotyledon angle predictions for every curated train/val image."""

import argparse
import csv
import glob
import os
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from cotyledon_angle.geometry import CotyledonPoints, cotyledon_angles
from cotyledon_angle.paths import DATASET_DIR, DEFAULT_EXPORT_CSV, FINAL_MODEL


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=FINAL_MODEL, help="YOLOv8 pose weights to use.")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=DATASET_DIR / "images",
        help="Directory containing train/ and val/ image folders.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPORT_CSV, help="CSV output path.")
    parser.add_argument("--conf", type=float, default=0.6, help="Prediction confidence threshold.")
    return parser.parse_args()


def run_split_analysis(model_path: Path, images_dir: Path, output_csv: Path, conf: float):
    """Predict angles for all images and write a compact CSV report."""
    model = YOLO(str(model_path))

    # Split identifies whether the image came from the train or validation set.
    header = ['Split', 'Filename', 'Stalk_Angle', 'Tip_Angle', 'Confidence_Avg']
    
    search_path = os.path.join(str(images_dir), "**", "*.jpg")
    image_paths = glob.glob(search_path, recursive=True)

    print(f"--- Processing {len(image_paths)} curated images from your 80/20 split ---")
    
    results_data = []

    for img_path in image_paths:
        results = model(img_path, conf=conf, verbose=False)
        
        split_name = os.path.basename(os.path.dirname(img_path))
        file_name = os.path.basename(img_path)

        for r in results:
            if r.keypoints is None or len(r.keypoints.xy[0]) < 5:
                results_data.append([split_name, file_name, "FAIL", "FAIL", 0.0])
                continue

            pts = r.keypoints.xy[0].cpu().numpy()
            keypoint_conf = r.keypoints.conf[0].cpu().numpy()
            cotyledon_points = CotyledonPoints.from_yolo(pts)
            stalk_ang, tip_ang = cotyledon_angles(cotyledon_points)

            results_data.append([
                split_name, 
                file_name, 
                round(stalk_ang, 2), 
                round(tip_ang, 2), 
                round(np.mean(keypoint_conf), 4)
            ])

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results_data)

    print(f"\nSUCCESS: Data for {len(image_paths)} images saved to {output_csv}")

if __name__ == "__main__":
    args = parse_args()
    run_split_analysis(args.model, args.images_dir, args.output, args.conf)
