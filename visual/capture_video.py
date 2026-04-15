import cv2
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.json_utils import log

def get_video_capture():
    """
    Initializes webcam securely using Windows stability settings.
    Returns: cv2.VideoCapture object
    """
    # Add CAP_DSHOW for Windows stability
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        log("Cannot open webcam")
        return None
        
    # Set to a highly supported and lightweight resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    return cap

