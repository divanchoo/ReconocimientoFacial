import cv2
import time
import numpy as np
from deepface import DeepFace

from src.detection.face_detector import detect_faces
from src.preprocess.align import crop_align_face
from src.models.embedding import get_embedding
from src.db.models import get_all_embeddings

# ---------- Config ----------
OVERLAY_ALPHA = 0.6
THRESHOLD = 0.55   # DeepFace distances ~0.3–0.7; menor = más seguro
MODEL_NAME = "Facenet512"
# ----------------------------


class Recognizer:
    def __init__(self):
        print(" Cargando base de datos de embeddings...")
        rows = get_all_embeddings()

        # rows: (id, name, embedding)
        self.names = [r[1] for r in rows]
        self.embs =  [r[2] for r in rows]

        if len(self.names) == 0:
            print("⚠ No hay usuarios registrados. Registra primero.")
            time.sleep(2)

        print(f" Usuarios en BD: {self.names}")

    # ------------------------------- UI Helpers -------------------------------
    def _draw_overlay_top(self, frame, text):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)
        cv2.putText(frame, text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    def _draw_overlay_bottom(self, frame, text):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)
        cv2.putText(frame, text, (10, h-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # --------------------------- Reconocimiento ------------------------------
    def _predict_identity(self, face_rgb):
        """
        face_rgb: rostro ya recortado y alineado (RGB)
        Devuelve (nombre, distancia)
        """

        if len(self.embs) == 0:
            return "BD Vacía", 999

        emb_new = get_embedding(face_rgb)
        if emb_new is None:
            return "ErrorEmbedding", 999

        distances = [np.linalg.norm(e - emb_new) for e in self.embs]
        idx = int(np.argmin(distances))
        min_dist = distances[idx]

        if min_dist <= THRESHOLD:
            return self.names[idx], float(min_dist)
        else:
            return "Desconocido", float(min_dist)

    # --------------------------- Loop Principal -------------------------------
    def start(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print(" No se pudo abrir la cámara.")
            return

        print("🎥 Reconocimiento iniciado. Presiona 'q' para salir.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_h, frame_w = frame.shape[:2]
            self._draw_overlay_top(frame, "Buscando rostro...")
            self._draw_overlay_bottom(frame, "Q: salir")

            # Detección con DeepFace (SSD)
            boxes = detect_faces(frame)

            if not boxes:
                # sin rostro
                cv2.rectangle(frame, (5,5), (frame_w-5, frame_h-5), (0,0,200), 2)
            else:
                for box in boxes:
                    (top, right, bottom, left) = box

                    # recorte + alineación
                    face_rgb = crop_align_face(frame, box)
                    if face_rgb is None:
                        continue

                    # predicción
                    nombre, dist = self._predict_identity(face_rgb)

                    # semáforo tamaño del rostro
                    w = right - left
                    face_ratio = w / float(frame_w)
                    if face_ratio >= 0.22:
                        border_color = (0,200,0)
                    elif face_ratio >= 0.12:
                        border_color = (0,200,200)
                    else:
                        border_color = (0,0,200)

                    # texto
                    color = (0,200,0) if nombre != "Desconocido" else (0,0,200)
                    text = f"{nombre} ({dist:.2f})" if nombre != "ErrorEmbedding" else "Error"

                    # rectángulo
                    cv2.rectangle(frame, (left, top), (right, bottom), border_color, 2)
                    cv2.putText(frame, text, (left, top-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

                    # overlays
                    if nombre != "Desconocido":
                        self._draw_overlay_top(frame, "Identificado")
                        cv2.putText(frame, f"Confianza: {dist:.2f}",
                                    (10, frame_h-60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (200,200,200), 2)
                    else:
                        self._draw_overlay_top(frame, "No identificado")

            cv2.imshow("Reconocimiento Facial", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Recognizer().start()
