"""
WEDA-CV Track Annotator — Drawing Primitives
============================================
Helpers called by the engine on every annotated frame:
  - get_color          : stable per-track colour from the palette
  - draw_track_labels  : "Track ID: N" labels above each box
  - draw_object_count  : "Objects Tracked: N" badge
Appearance is driven by the visualization block in config/config.yaml.
This file is part of the standard engine. Do not modify.
"""
import colorsys
import numpy as np
import cv2
# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
def _generate_palette(n: int = 64) -> list[tuple[int, int, int]]:
    palette = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        palette.append((int(b * 255), int(g * 255), int(r * 255)))
    return palette
PALETTE = _generate_palette()
def get_color(track_id: int) -> tuple[int, int, int]:
    return PALETTE[track_id % len(PALETTE)]
# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------
def draw_track_labels(
    frame: np.ndarray,
    xyxy: np.ndarray,
    track_ids: np.ndarray,
    confs: np.ndarray | None = None,
    show_conf: bool = False,
    font_scale: float = 0.6,
    thickness: int = 2,
) -> None:
    for i, tid in enumerate(track_ids):
        x1, y1 = int(xyxy[i][0]), int(xyxy[i][1])
        color = get_color(int(tid))
        label = f"Track ID: {int(tid)}"
        if show_conf and confs is not None:
            label += f" {confs[i]:.2f}"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        pad_x, pad_y = 6, 4
        lbl_y1 = max(y1 - th - 2 * pad_y - baseline, 0)
        lbl_y2 = y1
        lbl_x1 = x1
        lbl_x2 = x1 + tw + 2 * pad_x
        cv2.rectangle(frame, (lbl_x1, lbl_y1), (lbl_x2, lbl_y2), color, -1)
        cv2.putText(
            frame, label,
            (lbl_x1 + pad_x, lbl_y2 - pad_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255), thickness, cv2.LINE_AA,
        )

def draw_object_count(
    frame: np.ndarray,
    count: int,
    font_scale: float = 0.9,
    thickness: int = 2,
) -> None:
    text = f"Objects Tracked: {count}"
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
    )
    pad = 10
    cv2.rectangle(
        frame, (15, 15),
        (15 + tw + 2 * pad, 15 + th + baseline + 2 * pad),
        (0, 0, 0), -1,
    )
    cv2.putText(
        frame, text, (15 + pad, 15 + pad + th),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale,
        (255, 255, 255), thickness, cv2.LINE_AA,
    )
