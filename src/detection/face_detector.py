import cv2
import dlib

# Detector frontal de Dlib
detector = dlib.get_frontal_face_detector()

def detect_faces(frame):
    """
    Retorna lista de cajas (x, y, w, h) detectadas por Dlib.
    """

    # Convertir a gris si viene en BGR
    if len(frame.shape) == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    detections = detector(gray, 1)
    boxes = []

    for d in detections:
        x = d.left()
        y = d.top()
        w = d.right() - d.left()
        h = d.bottom() - d.top()
        boxes.append((x, y, w, h))

    return boxes
