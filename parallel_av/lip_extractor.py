"""
lip_extractor.py
=================
Minimal Lip-Region Extractor  (Parallel AV Module — Standalone)
-----------------------------------------------------------------
PURPOSE  : Utility to crop the lip ROI from a BGR video frame using
           MediaPipe Face Mesh landmarks.

USAGE    : Import `extract_lips()` from the parallel_av_pipeline or
           use standalone for quick testing::

               python lip_extractor.py

NOTES    :
  - This is a self-contained helper. It does NOT import app.py.
  - Designed for CPU-only, single-image processing (no GPU needed).
  - Returns a fixed 96×48 BGR crop or None if no face detected.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe Face Mesh — module-level singleton to avoid re-init overhead
# ──────────────────────────────────────────────────────────────────────────────

_mp_face_mesh = mp.solutions.face_mesh

# Lip landmark indices from MediaPipe Face Mesh (468-point model)
# Upper lip outer: 61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291
# Lower lip outer: 146, 91, 181, 84, 17, 314, 405, 321, 375, 291
# We use a minimum bounding set that captures the full mouth region.
LIP_LANDMARK_INDICES: tuple[int, ...] = (
    61,  185, 40,  39,  37,  0,   267, 269, 270, 409, 291,
    146, 91,  181, 84,  17,  314, 405, 321, 375,
)

# Output crop size (width × height)
LIP_CROP_W: int = 96
LIP_CROP_H: int = 48

# Padding around the tight bounding box (fraction of bbox size)
PADDING: float = 0.20


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def extract_lips(
    frame_bgr: np.ndarray,
    face_mesh: Optional[mp.solutions.face_mesh.FaceMesh] = None,
) -> Optional[np.ndarray]:
    """
    Detect the lip region in *frame_bgr* and return a fixed-size BGR crop.

    Parameters
    ----------
    frame_bgr : np.ndarray
        A BGR image from OpenCV (shape: H×W×3).
    face_mesh : mp.solutions.face_mesh.FaceMesh or None
        An already-initialised FaceMesh instance.
        If None, a temporary one is created (slightly slower per call).

    Returns
    -------
    np.ndarray  (96×48 BGR) if a face is detected.
    None        if no face found or frame is invalid.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    h, w = frame_bgr.shape[:2]

    # Convert BGR → RGB for MediaPipe
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # Use provided instance or create a temporary one
    _own_mesh = False
    if face_mesh is None:
        face_mesh = _mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
        )
        _own_mesh = True

    try:
        results = face_mesh.process(rgb)
    finally:
        if _own_mesh:
            face_mesh.close()

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark

    # ── Compute bounding box around lip landmarks ──
    xs = [landmarks[i].x * w for i in LIP_LANDMARK_INDICES]
    ys = [landmarks[i].y * h for i in LIP_LANDMARK_INDICES]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    bw = x_max - x_min
    bh = y_max - y_min

    # Apply padding
    pad_x = bw * PADDING
    pad_y = bh * PADDING

    x1 = max(0, int(x_min - pad_x))
    y1 = max(0, int(y_min - pad_y))
    x2 = min(w, int(x_max + pad_x))
    y2 = min(h, int(y_max + pad_y))

    # Guard against degenerate boxes
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Resize to fixed output dimensions
    lip_roi = cv2.resize(crop, (LIP_CROP_W, LIP_CROP_H), interpolation=cv2.INTER_AREA)
    return lip_roi


# ──────────────────────────────────────────────────────────────────────────────
# Quick standalone test
# ──────────────────────────────────────────────────────────────────────────────

def _quick_test() -> None:
    """
    Opens webcam, runs lip extraction on every captured frame, and
    shows both the full face and the extracted lip crop side-by-side.
    Press 'q' to exit.
    """
    print("[LipExtractor] Starting quick test — press 'q' to quit.")

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[LipExtractor] ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Reuse FaceMesh instance across frames for efficiency
    with _mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as fm:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            lip = extract_lips(frame, face_mesh=fm)

            if lip is not None:
                # Scale up for visibility in test window
                lip_display = cv2.resize(lip, (192, 96), interpolation=cv2.INTER_LINEAR)
                # Embed in top-right corner of frame
                frame[10 : 10 + 96, frame.shape[1] - 202 : frame.shape[1] - 10] = lip_display
                cv2.putText(
                    frame, "Lip ROI", (frame.shape[1] - 202, 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 80), 1, cv2.LINE_AA,
                )
            else:
                cv2.putText(
                    frame, "No face detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA,
                )

            cv2.imshow("Lip Extractor Test", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("[LipExtractor] Test complete.")


if __name__ == "__main__":
    _quick_test()
