import cv2
import numpy as np
from ultralytics import YOLO
import os
import random
import glob

# 1. Point to the FIXED model and the CLEANED validation data
MODEL_PATH = r'D:\rmproject\runs\pose\SaplingProject\v1_final_clean\weights\best.pt'
VAL_DIR = r'D:\rmproject\datasets\sapling_pose\images\val'

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

def run_clean_inference():
    # Load all images from your unseen 20% validation split
    all_val_imgs = glob.glob(os.path.join(VAL_DIR, "*.jpg"))
    
    if not all_val_imgs:
        print(f"Error: Could not find images in {VAL_DIR}")
        return

    # Pick 10 random images to showcase
    sample_imgs = random.sample(all_val_imgs, min(10, len(all_val_imgs)))
    print(f"--- Running Inference on {len(sample_imgs)} Cleaned Validation Images ---")
    print("Press any key to jump to the next image. Press 'q' to quit early.")

    for img_path in sample_imgs:
        # Run the final trained model
        results = model(img_path, conf=0.6, verbose=False)
        
        for r in results:
            img = r.orig_img.copy()
            
            if r.keypoints is None or len(r.keypoints.xy[0]) < 5:
                print(f"Skipping {os.path.basename(img_path)}: 5 points not detected.")
                continue

            # Extract the points predicted by the model
            pts = r.keypoints.xy[0].cpu().numpy()
            
            # MAPPING: Junction is the 5th click (Index 4)
            p_tip_l, p_base_l = pts[0], pts[1]
            p_base_r, p_tip_r = pts[2], pts[3]
            junc = pts[4] 

            # Calculate the scientific angles
            stalk_ang = get_angle(p_base_l, junc, p_base_r)
            tip_ang = get_angle(p_tip_l, junc, p_tip_r)

            # ---------------------------------------------------------
            # VISUALIZATION (Matching your screenshots exactly)
            # ---------------------------------------------------------
            
            # 1. Draw Skeleton (Green Lines)
            skeleton = [(p_tip_l, p_base_l), (p_base_l, junc), (junc, p_base_r), (p_base_r, p_tip_r)]
            for start, end in skeleton:
                cv2.line(img, tuple(start.astype(int)), tuple(end.astype(int)), (0, 255, 0), 2)

            # 2. Draw Points (Red=Tips, Blue=Bases, Yellow=Junction)
            colors = [(0, 0, 255), (255, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255)]
            for i, p in enumerate(pts):
                cv2.circle(img, tuple(p.astype(int)), 5, colors[i], -1)

            # 3. Add the Black Header with Green Text
            cv2.rectangle(img, (0, 0), (320, 100), (0, 0, 0), -1)
            cv2.putText(img, f"Stalk Angle: {stalk_ang:.2f} deg", (10, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Tip Angle:   {tip_ang:.2f} deg", (10, 80), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Show the result
            cv2.imshow("Fixed YOLO Prediction", img)
            print(f"Showing: {os.path.basename(img_path)}")
            
            # Wait for key press
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                return
                
    cv2.destroyAllWindows()
    print("Done showing the 10 showcase images!")

if __name__ == "__main__":
    run_clean_inference()