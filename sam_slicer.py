import cv2
import os
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
        
    # Step 1: Pre-processing
    # Convert to grayscale and apply Gaussian Blur to remove sensor noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 2: Thresholding (Otsu's Method)
    # This automatically finds the best background/foreground cut-off
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 3: Morphological Closing
    # This 'glues' the thin stems and leaves together so they are treated as one blob
    kernel = np.ones((7, 7), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Step 4: Contour Detection
    # Find the external boundaries of the plants
    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    crop_paths = []
    print(f"   [CV-Engine] Analyzing {os.path.basename(image_path)}...")

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        
        # Step 5: Area Filtering (The 'Big Enough' check)
        # 1000 pixels is usually safe for a whole sapling, adjust if needed
        if area < 1000:
            continue
            
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Step 6: Padding & Cropping
        # We add 30px padding so the leaf tips aren't cut off for the Analyst Agent
        pad = 30
        y1, y2 = max(0, y-pad), min(image.shape[0], y+h+pad)
        x1, x2 = max(0, x-pad), min(image.shape[1], x+w+pad)
        
        crop = image[y1:y2, x1:x2]
        
        crop_filename = os.path.join(output_subfolder, f"sapling_{i}.jpg")
        cv2.imwrite(crop_filename, crop)
        crop_paths.append(crop_filename)
        
    print(f"   [CV-Engine] Successfully isolated {len(crop_paths)} whole plants.")
    return crop_paths