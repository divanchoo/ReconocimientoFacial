import cv2

class FaceDetector:
    def __init__(self, scaleFactor=1.3, minNeighbors=5):
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.scaleFactor = scaleFactor
        self.minNeighbors = minNeighbors

    def detect(self, gray_frame):
        """
        Recibe un frame en escala de grises y devuelve:
        - lista de bounding boxes (x, y, w, h)
        """
        faces = self.detector.detectMultiScale(
            gray_frame,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors
        )
        return faces
