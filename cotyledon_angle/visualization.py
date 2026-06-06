"""OpenCV drawing helpers for cotyledon landmark predictions."""

import cv2

from cotyledon_angle.geometry import CotyledonPoints


YOLO_POINT_COLORS = [
    (0, 0, 255),      # left tip: red
    (255, 0, 0),      # left base: blue
    (255, 0, 0),      # right base: blue
    (0, 0, 255),      # right tip: red
    (0, 255, 255),    # junction: yellow
]


def draw_yolo_measurement(
    image,
    points: CotyledonPoints,
    stalk_angle: float,
    tip_angle: float,
    *,
    header: bool = True,
):
    """Draw the botanical skeleton and angle readout on an OpenCV image."""
    pts = [point.astype(int) for point in points.as_yolo_order]

    # Skeleton follows leaf tip -> leaf base -> junction -> leaf base -> leaf tip.
    skeleton = [(pts[0], pts[1]), (pts[1], pts[4]), (pts[4], pts[2]), (pts[2], pts[3])]
    for start, end in skeleton:
        cv2.line(image, tuple(start), tuple(end), (0, 255, 0), 2)

    for idx, point in enumerate(pts):
        cv2.circle(image, tuple(point), 5, YOLO_POINT_COLORS[idx], -1)
        cv2.putText(image, str(idx), tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if header:
        cv2.rectangle(image, (0, 0), (340, 100), (0, 0, 0), -1)
        cv2.putText(
            image,
            f"Stalk Angle: {stalk_angle:.2f} deg",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            image,
            f"Tip Angle:   {tip_angle:.2f} deg",
            (10, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return image
