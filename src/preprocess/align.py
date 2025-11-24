import cv2
import numpy as np

def crop_align_face(frame, box, target_size=(160,160)):
    """
    box: (top, right, bottom, left)
    Returns: aligned/resized RGB image (numpy array) ready for embedding extraction.
    """
    top, right, bottom, left = box
    h, w = frame.shape[:2]
    # ensure bounds
    top = max(0, top)
    left = max(0, left)
    bottom = min(h, bottom)
    right = min(w, right)
    face = frame[top:bottom, left:right]
    if face.size == 0:
        return None
    # convert to RGB
    face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face_resized = cv2.resize(face_rgb, target_size)
    return face_resized
