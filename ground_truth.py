"""Visual checker for recovered manual keypoint annotations."""

import argparse
import json
import os
import random
from pathlib import Path

import cv2


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=Path("recovered_points.json"), help="Recovered manual points JSON.")
    parser.add_argument("--source-dir", type=Path, default=Path("Output"), help="Root folder containing crop images.")
    parser.add_argument("--samples", type=int, default=5, help="Number of random annotations to inspect.")
    return parser.parse_args()


def verify_ground_truth(json_path: Path, source_dir: Path, samples: int):
    """Overlay manual clicks on crops so annotation quality can be checked."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Entries stored as strings were skipped during recovery; dicts contain the
    # crop path plus five manual cotyledon points.
    valid_keys = [k for k, v in data.items() if isinstance(v, dict)]
    if not valid_keys:
        print("No valid manual annotations found.")
        return

    for key in random.sample(valid_keys, min(samples, len(valid_keys))):
        entry = data[key]
        img_path = source_dir / entry['parent_image'] / entry['crop']

        img = cv2.imread(os.fspath(img_path))
        if img is None:
            print(f"Skipping missing crop: {img_path}")
            continue

        # Numeric labels reveal point ordering mistakes immediately.
        for i, pt in enumerate(entry['points']):
            cv2.circle(img, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)
            cv2.putText(img, str(i), (int(pt[0]), int(pt[1])), 0, 0.5, (255,255,255), 1)

        cv2.imshow("JSON DATA VERIFICATION", img)
        print(f"Checking: {key} | If dots are floating here, the recovered JSON is wrong.")
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    args = parse_args()
    verify_ground_truth(args.json, args.source_dir, args.samples)
