import cv2
import pickle
import time
from src.detection.face_detector import FaceDetector

# ---------- Config ----------
OVERLAY_ALPHA = 0.6
THRESHOLD = 70  # menor = más seguro en LBPH: aquí consideramos < THRESHOLD como aceptable
# ----------------------------

class Recognizer:
    def __init__(self):
        self.model_path = "src/data/model.xml"
        self.labels_path = "src/data/labels.pickle"

        # Cargar modelo LBPH
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
            self.recognizer.read(self.model_path)
        except Exception as e:
            print(" No se encontró el modelo. Entrena primero.")
            raise e

        # Cargar labels
        try:
            with open(self.labels_path, "rb") as f:
                self.label_dict = pickle.load(f)
        except Exception as e:
            print(" No se encontró labels.pickle. Entrena primero.")
            raise e

        self.detector = FaceDetector()

    def _draw_overlay_top(self, frame, text):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)
        cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    def _draw_overlay_bottom(self, frame, text):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)
        cv2.putText(frame, text, (10, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

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
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detect(gray)

            # Mensajes por defecto
            self._draw_overlay_top(frame, "Buscando rostro...")
            self._draw_overlay_bottom(frame, "Q: salir")

            if len(faces) == 0:
                # sin rostro
                cv2.rectangle(frame, (5,5), (frame_w-5, frame_h-5), (0,0,200), 2)
            else:
                for (x, y, w, h) in faces:
                    rostro = gray[y:y+h, x:x+w]
                    try:
                        label, confidence = self.recognizer.predict(rostro)
                    except:
                        label, confidence = -1, 999

                    # decidir color y texto según threshold
                    if confidence < THRESHOLD:
                        nombre = self.label_dict.get(label, "Desconocido")
                        color = (0, 200, 0)
                        text = f"{nombre} ({int(confidence)})"
                    else:
                        nombre = "Desconocido"
                        color = (0, 0, 200)
                        text = "Desconocido"

                    # semáforo por calidad (tamaño relativo)
                    face_ratio = w / float(frame_w)
                    if face_ratio >= 0.22:
                        # buena
                        border_color = (0,200,0)
                    elif face_ratio >= 0.12:
                        border_color = (0,200,200)
                    else:
                        border_color = (0,0,200)

                    # rectángulo y texto
                    cv2.rectangle(frame, (x, y), (x+w, y+h), border_color, 2)
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # mensaje superior con detalle
                    self._draw_overlay_top(frame, f"{'Identificado' if nombre!='Desconocido' else 'No identificado'}")

                    # mostrar confianza si es buena (opcional, minimal)
                    if confidence < THRESHOLD:
                        cv2.putText(frame, f"Confianza: {int(confidence)}", (10, frame_h-60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,200), 2)

            cv2.imshow("Reconocimiento Facial", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    Recognizer().start()
