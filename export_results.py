import cv2
import numpy as np
from ultralytics import YOLO
import os
import csv
import glob

# --- Configuration ---
MODEL_PATH = r'D:\rmproject\runs\pose\SaplingProject\v1_final_clean\weights\best.pt'
DATASET_DIR = r'D:\rmproject\datasets\sapling_pose\images'
OUTPUT_CSV = 'sapling_cleaned_data.csv'

model = YOLO(MODEL_PATH)

def get_angle(A, B, C):
    """Calculates angle ABC (vertex at B) using vector dot product."""
    ba = A - B
    bc = C - B
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0: return 0
    
    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def run_split_analysis():
    # Header: Split identifies if it was from the 80% train or 20% val set
    header = ['Split', 'Filename', 'Stalk_Angle', 'Tip_Angle', 'Confidence_Avg']
    
    # Target only the images YOLO is actually using
    search_path = os.path.join(DATASET_DIR, "**", "*.jpg")
    image_paths = glob.glob(search_path, recursive=True)

    print(f"--- Processing {len(image_paths)} curated images from your 80/20 split ---")
    
    results_data = []

    for img_path in image_paths:
        # Run model
        results = model(img_path, conf=0.6, verbose=False)
        
        # Identify if it's 'train' or 'val' folder
        split_name = os.path.basename(os.path.dirname(img_path))
        file_name = os.path.basename(img_path)

        for r in results:
            if r.keypoints is None or len(r.keypoints.xy[0]) < 5:
                results_data.append([split_name, file_name, "FAIL", "FAIL", 0.0])
                continue

            # Junction is at index 4 (last click)
            pts = r.keypoints.xy[0].cpu().numpy()
            conf = r.keypoints.conf[0].cpu().numpy()
            
            p_tip_l, p_base_l = pts[0], pts[1]
            p_base_r, p_tip_r = pts[2], pts[3]
            junc = pts[4] 

            # Vectorized Angle Calculation
            stalk_ang = get_angle(p_base_l, junc, p_base_r)
            tip_ang = get_angle(p_tip_l, junc, p_tip_r)

            results_data.append([
                split_name, 
                file_name, 
                round(stalk_ang, 2), 
                round(tip_ang, 2), 
                round(np.mean(conf), 4)
            ])

    # Save to CSV
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(results_data)

    print(f"\nSUCCESS: Data for {len(image_paths)} images saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_split_analysis()