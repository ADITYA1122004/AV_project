import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False, 
    max_num_faces=1, 
    refine_landmarks=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Indices for lips in mediapipe
LIPS_INDICES = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
    191, 80, 81, 82, 13, 312, 311, 310, 415, 308
]

def extract_lips(frame):
    """
    Extracts the lip region, resizes to 96x96, and converts to grayscale, normalizing.
    Returns: (processed_frame, bounding_box) or None
    """
    # 1. Convert frame
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 2. Run MediaPipe
    results = face_mesh.process(frame_rgb)
    
    if not results.multi_face_landmarks:
        return None
        
    landmarks = results.multi_face_landmarks[0].landmark
    h, w, _ = frame.shape
    
    # 3. Extract lip points
    x_coords = []
    y_coords = []
    
    for idx in LIPS_INDICES:
        point = landmarks[idx]
        x_coords.append(int(point.x * w))
        y_coords.append(int(point.y * h))
        
    # 4. Compute bounding box
    x_min, x_max = max(0, min(x_coords) - 10), min(w, max(x_coords) + 10)
    y_min, y_max = max(0, min(y_coords) - 10), min(h, max(y_coords) + 10)
    
    if x_min >= x_max or y_min >= y_max or x_min < 0 or y_min < 0 or x_max > w or y_max > h:
        return None
        
    # 5. Crop ROI
    lip_roi = frame[y_min:y_max, x_min:x_max]
    
    if lip_roi.size == 0:
        return None
        
    # Frame Preprocessing: Resize -> Grayscale -> Normalize
    lip_resized = cv2.resize(lip_roi, (96, 96))
    lip_gray = cv2.cvtColor(lip_resized, cv2.COLOR_BGR2GRAY)
    lip_norm = lip_gray.astype(np.float32) / 255.0
    
    return lip_norm, (x_min, y_min, x_max, y_max)

