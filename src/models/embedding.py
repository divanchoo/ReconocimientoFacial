# src/models/embedding.py
from deepface import DeepFace
import numpy as np

def get_embedding(face_rgb):
    """
    Devuelve un embedding usando Facenet512.
    """
    try:
        emb = DeepFace.represent(
            face_rgb,
            model_name="Facenet512",
            enforce_detection=False
        )[0]["embedding"]
        return np.array(emb, dtype="float32")
    except:
        return None
