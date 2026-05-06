import cv2
import numpy as np
from ultralytics import YOLO
import os
import glob

MODEL_PATH = r'D:\rmproject\runs\pose\SaplingProject\v1_manual_points-4\weights\best.pt'
VAL_PATH = r'D:\rmproject\datasets\sapling_pose\images\val'
model = YOLO(MODEL_PATH)

def calculate_angle(A, B, C):
    ba = A - B
    bc = C - B
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

def test_val():
    val_imgs = glob.glob(os.path.join(VAL_PATH, "*.jpg"))
    for img_path in val_imgs:
        results = model(img_path, conf=0.4)
        for r in results:
            img = r.orig_img.copy()
            if r.keypoints is None or len(r.keypoints.xy[0]) < 5: continue

            pts = r.keypoints.xy[0].cpu().numpy()
            
            # CORRECT MAPPING BASED ON YOUR CLICKS:
            # You said Junction was LAST (Index 4)
            p0, p1, p2, p3 = pts[0], pts[1], pts[2], pts[3]
            junc = pts[4] 

            # Calculate angles using Point 4 as the vertex
            stalk_ang = calculate_angle(p1, junc, p2) # Assuming inner points are 1 & 2
            tip_ang = calculate_angle(p0, junc, p3)   # Assuming outer points are 0 & 3

            # Draw "Skeleton"
            for i, p in enumerate(pts):
                cv2.circle(img, tuple(p.astype(int)), 5, (255, 0, 0), -1)
                cv2.putText(img, str(i), tuple(p.astype(int)), 0, 0.6, (255,255,255), 2)
                if i < 4: # Draw lines to the junction
                    cv2.line(img, tuple(p.astype(int)), tuple(junc.astype(int)), (0, 255, 0), 1)

            # Display Stats
            cv2.putText(img, f"Stalk: {stalk_ang:.1f} | Tip: {tip_ang:.1f}", (10, 30), 0, 0.7, (0, 255, 0), 2)
            cv2.imshow("Validation Result", img)
            if cv2.waitKey(0) & 0xFF == ord('q'): return

if __name__ == "__main__":
    test_val()