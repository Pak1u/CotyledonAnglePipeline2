import cv2
import numpy as np
import json
import base64
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv

load_dotenv()
# Initialize your AI - Using the Pro model for maximum coordinate precision
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview", # High-fidelity vision with low latency
    google_api_key=GEMINI_API_KEY,
    temperature=0 
)

def calculate_angle(p1, vertex, p2):
    r"""
    Computes the interior angle at the vertex using:
    $$\theta = \arccos\left(\frac{\vec{ba} \cdot \vec{bc}}{|\vec{ba}| \cdot |\vec{bc}|}\right)$$
    """
    ba = np.array(p1) - np.array(vertex)
    bc = np.array(p2) - np.array(vertex)
    
    dot_product = np.dot(ba, bc)
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = dot_product / (norm_ba * norm_bc)
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return round(angle, 2)

def analyze_sapling(crop_path):
    """
    Analyzes a sapling crop using a 5-point botanical map.
    Calculates both Stalk Angle and Tip Angle.
    """
    try:
        img_orig = cv2.imread(crop_path)
        if img_orig is None: return "NON_PLANT"
        orig_h, orig_w = img_orig.shape[:2]

        # STEP 1: Aspect Ratio Preservation (Fixes point distortion)
        scaling_factor = 1000.0 / max(orig_h, orig_w)
        target_w = int(orig_w * scaling_factor)
        target_h = int(orig_h * scaling_factor)
        img_resized = cv2.resize(img_orig, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        _, buffer = cv2.imencode(".jpg", img_resized)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        # STEP 2: The Master Prompt
        prompt = (
            "You are a botanical vision expert. Analyze this image. "
            "If this is NOT a valid dicot sapling, return 'INVALID'. "
            "If valid, identify these [x, y] coordinates on a 0-1000 scale: "
            "1. Sharp tip of the LEFT cotyledon. "
            "2. Base of the LEFT cotyledon (where the stalk meets the leaf blade). "
            "3. Central junction (vertex) where both stalks meet the stem. "
            "4. Base of the RIGHT cotyledon (where the stalk meets the leaf blade). "
            "5. Sharp tip of the RIGHT cotyledon. "
            "Output ONLY JSON: [[x1, y1], [x2, y2], [vx, vy], [x4, y4], [x5, y5]]."
        )
        
        message = HumanMessage(content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
        ])
        
        response = llm.invoke([message])

        # STEP 3: Robust Content Extraction (Fixes 'list' object error)
        raw_content = response.content
        if isinstance(raw_content, list):
            content = "".join([part["text"] if isinstance(part, dict) else str(part) for part in raw_content])
        else:
            content = str(raw_content)
        
        content = content.strip()

        if "INVALID" in content.upper():
            return "NON_PLANT"

        # JSON Cleaning
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        norm_coords = json.loads(content.strip())
        
        if not isinstance(norm_coords, list) or len(norm_coords) != 5:
            return "NON_PLANT"

        # STEP 4: Re-map Coordinates to Original Resolution
        # Mapping Gemini's 0-1000 back to the original image dimensions
        real_coords = []
        for x_n, y_n in norm_coords:
            px_x = int((x_n / 1000) * orig_w)
            px_y = int((y_n / 1000) * orig_h)
            real_coords.append([px_x, px_y])

        # STEP 5: Calculate Dual Angles
        # Points: 0:L_Tip, 1:L_Base, 2:Junction, 3:R_Base, 4:R_Tip
        stalk_angle = calculate_angle(real_coords[1], real_coords[2], real_coords[3])
        tip_angle = calculate_angle(real_coords[0], real_coords[2], real_coords[4])
        
        _draw_result(img_orig, crop_path, real_coords, stalk_angle, tip_angle)
        
        return {"stalk_angle": stalk_angle, "tip_angle": tip_angle}

    except Exception as e:
        error_msg = str(e).upper()
        if any(code in error_msg for code in ["503", "504", "429", "UNAVAILABLE", "EXHAUSTED"]):
            raise e
        print(f"      [!] Analysis failed for {os.path.basename(crop_path)}: {e}")
        return "NON_PLANT"

def _draw_result(img, path, coords, s_angle, t_angle):
    """Draws both stalk and tip measurements."""
    # Points: 0:L_Tip, 1:L_Base, 2:Junction, 3:R_Base, 4:R_Tip
    pts = [tuple(map(int, c)) for c in coords]
    
    # Draw Stalk Angle (Yellow)
    cv2.line(img, pts[2], pts[1], (0, 255, 255), 2)
    cv2.line(img, pts[2], pts[3], (0, 255, 255), 2)
    
    # Draw Tip Angle (Green)
    cv2.line(img, pts[2], pts[0], (0, 255, 0), 1)
    cv2.line(img, pts[2], pts[4], (0, 255, 0), 1)
    
    # Mark all 5 points
    for i, pt in enumerate(pts):
        color = (0, 0, 255) if i == 2 else (255, 0, 0)
        cv2.circle(img, pt, 5, color, -1)
    
    # UI Text
    cv2.putText(img, f"Stalk: {s_angle} deg", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(img, f"Tip: {t_angle} deg", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    result_path = path.replace(".jpg", "_measured.jpg")
    cv2.imwrite(result_path, img)