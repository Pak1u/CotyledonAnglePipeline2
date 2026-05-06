import cv2
import json
import os
import random

# --- CONFIG ---
JSON_PATH = "recovered_points.json"
SOURCE_DIR = "Output"

with open(JSON_PATH, 'r') as f:
    data = json.load(f)

# Pick 5 random images that aren't "SKIPPED"
valid_keys = [k for k, v in data.items() if isinstance(v, dict)]
samples = random.sample(valid_keys, 5)

for key in samples:
    entry = data[key]
    img_path = os.path.join(SOURCE_DIR, entry['parent_image'], entry['crop'])
    
    img = cv2.imread(img_path)
    if img is None: continue
    
    # Draw your manual clicks from the JSON
    for i, pt in enumerate(entry['points']):
        cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
        cv2.putText(img, str(i), (int(pt[0]), int(pt[1])), 0, 0.5, (255,255,255), 1)

    cv2.imshow("JSON DATA VERIFICATION", img)
    print(f"Checking: {key} | If dots are floating here, your JSON is wrong.")
    cv2.waitKey(0)
cv2.destroyAllWindows()