"""Geometry primitives used by both YOLO and vision-model workflows."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CotyledonPoints:
    """Named view of the five botanical landmarks used by the pipeline.

    YOLO labels in this project are ordered as:
    0 left cotyledon tip, 1 left cotyledon base, 2 right cotyledon base,
    3 right cotyledon tip, 4 central junction.
    """

    left_tip: np.ndarray
    left_base: np.ndarray
    right_base: np.ndarray
    right_tip: np.ndarray
    junction: np.ndarray

    @classmethod
    def from_yolo(cls, points: np.ndarray) -> "CotyledonPoints":
        """Build named landmarks from a YOLOv8 pose keypoint array."""
        if len(points) < 5:
            raise ValueError("Expected at least 5 cotyledon keypoints")

        return cls(
            left_tip=np.asarray(points[0], dtype=float),
            left_base=np.asarray(points[1], dtype=float),
            right_base=np.asarray(points[2], dtype=float),
            right_tip=np.asarray(points[3], dtype=float),
            junction=np.asarray(points[4], dtype=float),
        )

    @classmethod
    def from_gemini(cls, points: list[list[int]]) -> "CotyledonPoints":
        """Build named landmarks from the legacy Gemini prompt order.

        Gemini was prompted as left tip, left base, junction, right base,
        right tip. Keeping this separate avoids silently swapping points.
        """
        if len(points) != 5:
            raise ValueError("Expected exactly 5 Gemini cotyledon points")

        return cls(
            left_tip=np.asarray(points[0], dtype=float),
            left_base=np.asarray(points[1], dtype=float),
            right_base=np.asarray(points[3], dtype=float),
            right_tip=np.asarray(points[4], dtype=float),
            junction=np.asarray(points[2], dtype=float),
        )

    @property
    def as_yolo_order(self) -> list[np.ndarray]:
        """Return points in the trained YOLO label order for drawing."""
        return [
            self.left_tip,
            self.left_base,
            self.right_base,
            self.right_tip,
            self.junction,
        ]


def calculate_angle(point_a: np.ndarray, vertex: np.ndarray, point_c: np.ndarray) -> float:
    """Return the interior angle A-vertex-C in degrees.

    A zero-length arm can happen when a model predicts two landmarks at the
    same pixel. Returning 0.0 keeps batch exports alive while making failures
    obvious in the output CSV.
    """
    ba = np.asarray(point_a, dtype=float) - np.asarray(vertex, dtype=float)
    bc = np.asarray(point_c, dtype=float) - np.asarray(vertex, dtype=float)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return float(np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0))))


def cotyledon_angles(points: CotyledonPoints) -> tuple[float, float]:
    """Measure stalk and tip spread from named cotyledon landmarks."""
    stalk_angle = calculate_angle(points.left_base, points.junction, points.right_base)
    tip_angle = calculate_angle(points.left_tip, points.junction, points.right_tip)
    return stalk_angle, tip_angle

