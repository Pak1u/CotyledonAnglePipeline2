"""Repository-relative paths used by the command-line scripts."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "data.yaml"
DATASET_DIR = PROJECT_ROOT / "datasets" / "sapling_pose"
FINAL_RUN_DIR = PROJECT_ROOT / "runs" / "pose" / "SaplingProject" / "v1_final_clean"
FINAL_MODEL = FINAL_RUN_DIR / "weights" / "best.pt"
BASE_POSE_MODEL = PROJECT_ROOT / "yolov8n-pose.pt"
DEFAULT_EXPORT_CSV = PROJECT_ROOT / "sapling_cleaned_data.csv"

