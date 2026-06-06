"""Segment full-tray images into individual sapling crops.

Despite the historical file name, this uses deterministic OpenCV thresholding
instead of Segment Anything. It supports the legacy Gemini measuring workflow.
"""

import os

import cv2
import numpy as np

def segment_and_save_crops(image_path, output_subfolder):
    """
    Uses traditional CV (Otsu Thresholding + Morphology) to isolate whole plants.
    Fast, deterministic, and avoids the over-segmentation issues of SAM.
    """
    if not os.path.exists(output_subfolder):
        os.makedirs(output_subfolder)

    # Load original color image
    image = cv2.imread(image_path)
    if image is None:
        print(f"   [Error] Could not read image at: {image_path}")
        return []
        
    # Convert to grayscale and blur first so Otsu thresholding is less sensitive
    # to sensor noise and tiny soil/background texture.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu chooses the foreground/background cut-off from each image histogram.
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Closing joins thin stems/leaves into one contour so each plant becomes a
    # single crop candidate.
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Only external contours are needed because each crop should contain a whole
    # sapling, not internal leaf/stem holes.
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crop_paths = []
    print(f"   [CV-Engine] Analyzing {os.path.basename(image_path)}...")

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Reject tiny specks; 1000 px was chosen for the original tray images.
        if area < 1000:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Padding protects leaf tips from being clipped by the contour rectangle.
        pad = 30
        y1, y2 = max(0, y-pad), min(image.shape[0], y+h+pad)
        x1, x2 = max(0, x-pad), min(image.shape[1], x+w+pad)
        
        crop = image[y1:y2, x1:x2]
        
        crop_filename = os.path.join(output_subfolder, f"sapling_{i}.jpg")
        cv2.imwrite(crop_filename, crop)
        crop_paths.append(crop_filename)
        
    print(f"   [CV-Engine] Successfully isolated {len(crop_paths)} whole plants.")
    return crop_paths
