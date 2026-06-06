"""Legacy full-validation viewer kept for comparing old/manual-point runs."""

import glob
import os
from pathlib import Path

import cv2
from ultralytics import YOLO

from cotyledon_angle.geometry import CotyledonPoints, cotyledon_angles
from cotyledon_angle.paths import DATASET_DIR, PROJECT_ROOT
from cotyledon_angle.visualization import draw_yolo_measurement

MODEL_PATH = PROJECT_ROOT / "runs" / "pose" / "SaplingProject" / "v1_manual_points-4" / "weights" / "best.pt"
VAL_PATH = DATASET_DIR / "images" / "val"

def test_val():
    """Display every validation image for visual debugging of older weights."""
    model = YOLO(str(MODEL_PATH))
    val_imgs = glob.glob(os.path.join(str(VAL_PATH), "*.jpg"))

    for img_path in val_imgs:
        results = model(img_path, conf=0.4)
        for r in results:
            img = r.orig_img.copy()
            if r.keypoints is None or len(r.keypoints.xy[0]) < 5:
                continue

            pts = r.keypoints.xy[0].cpu().numpy()
            cotyledon_points = CotyledonPoints.from_yolo(pts)
            stalk_ang, tip_ang = cotyledon_angles(cotyledon_points)
            draw_yolo_measurement(img, cotyledon_points, stalk_ang, tip_ang)

            cv2.imshow("Validation Result", img)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                return

if __name__ == "__main__":
    test_val()
