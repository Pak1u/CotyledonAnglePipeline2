"""Legacy Gemini-based cotyledon measurer for cropped sapling images.

The primary project path is YOLOv8 pose. This module is kept as an optional
fallback/experiment for cases where cropped plants are measured by a vision LLM.
"""

import base64
import json
import os
from pathlib import Path

import cv2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

from cotyledon_angle.geometry import CotyledonPoints, cotyledon_angles

load_dotenv()

# Gemini is only initialized once; analyze_sapling is called repeatedly by
# crop-processing scripts.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    google_api_key=GEMINI_API_KEY,
    temperature=0 
)


def analyze_sapling(crop_path):
    """Measure one cropped sapling with a Gemini-generated 5-point map."""
    try:
        img_orig = cv2.imread(crop_path)
        if img_orig is None:
            return "NON_PLANT"

        orig_h, orig_w = img_orig.shape[:2]

        # Gemini receives a normalized image, but aspect ratio is preserved so
        # its 0-1000 coordinate output can be mapped back without distortion.
        scaling_factor = 1000.0 / max(orig_h, orig_w)
        target_w = int(orig_w * scaling_factor)
        target_h = int(orig_h * scaling_factor)
        img_resized = cv2.resize(img_orig, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        
        _, buffer = cv2.imencode(".jpg", img_resized)
        img_b64 = base64.b64encode(buffer).decode("utf-8")

        # Prompt order is intentionally different from YOLO label order:
        # left tip, left base, junction, right base, right tip.
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

        # LangChain providers can return either a string or a list of parts.
        raw_content = response.content
        if isinstance(raw_content, list):
            content = "".join([part["text"] if isinstance(part, dict) else str(part) for part in raw_content])
        else:
            content = str(raw_content)
        
        content = content.strip()

        if "INVALID" in content.upper():
            return "NON_PLANT"

        # Strip fenced-code formatting if the model wrapped its JSON.
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        norm_coords = json.loads(content.strip())
        
        if not isinstance(norm_coords, list) or len(norm_coords) != 5:
            return "NON_PLANT"

        # Map Gemini's 0-1000 coordinates back to original image pixels.
        real_coords = []
        for x_n, y_n in norm_coords:
            px_x = int((x_n / 1000) * orig_w)
            px_y = int((y_n / 1000) * orig_h)
            real_coords.append([px_x, px_y])

        cotyledon_points = CotyledonPoints.from_gemini(real_coords)
        stalk_angle, tip_angle = [round(angle, 2) for angle in cotyledon_angles(cotyledon_points)]
        
        _draw_result(img_orig, Path(crop_path), cotyledon_points, stalk_angle, tip_angle)
        
        return {"stalk_angle": stalk_angle, "tip_angle": tip_angle}

    except Exception as e:
        error_msg = str(e).upper()
        if any(code in error_msg for code in ["503", "504", "429", "UNAVAILABLE", "EXHAUSTED"]):
            raise e
        print(f"      [!] Analysis failed for {os.path.basename(crop_path)}: {e}")
        return "NON_PLANT"

def _draw_result(img, path: Path, points: CotyledonPoints, s_angle, t_angle):
    """Draw Gemini-derived stalk and tip measurements onto the crop."""
    pts = {
        "left_tip": tuple(points.left_tip.astype(int)),
        "left_base": tuple(points.left_base.astype(int)),
        "right_base": tuple(points.right_base.astype(int)),
        "right_tip": tuple(points.right_tip.astype(int)),
        "junction": tuple(points.junction.astype(int)),
    }
    
    # Stalk angle uses the two cotyledon bases and central junction.
    cv2.line(img, pts["junction"], pts["left_base"], (0, 255, 255), 2)
    cv2.line(img, pts["junction"], pts["right_base"], (0, 255, 255), 2)
    
    # Tip angle uses the outer cotyledon tips and central junction.
    cv2.line(img, pts["junction"], pts["left_tip"], (0, 255, 0), 1)
    cv2.line(img, pts["junction"], pts["right_tip"], (0, 255, 0), 1)
    
    for name, pt in pts.items():
        color = (0, 0, 255) if name == "junction" else (255, 0, 0)
        cv2.circle(img, pt, 5, color, -1)
    
    cv2.putText(img, f"Stalk: {s_angle} deg", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(img, f"Tip: {t_angle} deg", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    result_path = path.with_name(f"{path.stem}_measured{path.suffix}")
    cv2.imwrite(str(result_path), img)
