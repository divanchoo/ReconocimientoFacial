import cv2
import pickle
import sys, os
from tkinter import Tk, messagebox
from src.detection.face_detector import FaceDetector

# ---------- Config ----------
OVERLAY_ALPHA = 0.6
# CAMBIO IMPORTANTE: LBPH devuelve "distancia" (0 es idéntico). 
# 70 es muy estricto. 110 es un buen punto de partida.
THRESHOLD = 65  
# ----------------------------

def resource_path(relative_path):
    """Obtiene rutas reales tanto en desarrollo como dentro del .exe PyInstaller."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


class Recognizer:
    def __init__(self):
        self.model_path = resource_path("src/data/model.xml")
        self.labels_path = resource_path("src/data/labels.pickle")

        # Tkinter oculto para mostrar mensajes
        root = Tk()
        root.withdraw()

        # Cargar modelo LBPH
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
            self.recognizer.read(self.model_path)
        except Exception:
            messagebox.showerror("Error", "No se encontró el modelo.\nEntrena primero.")
            sys.exit()

        # Cargar labels
        try:
            with open(self.labels_path, "rb") as f:
                self.label_dict = pickle.load(f)
        except Exception:
            messagebox.showerror("Error", "No se encontró labels.pickle.\nEntrena primero.")
            sys.exit()

        self.detector = FaceDetector()


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

    def start(self):
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # CAP_DSHOW ayuda a iniciar rápido en Windows
        if not cap.isOpened():
            messagebox.showerror("Error", "No se pudo abrir la cámara.")
            return

        messagebox.showinfo(
            "Reconocimiento iniciado",
            "La cámara está activa.\nMira la consola para ver los valores de confianza.\nPresiona 'Q' para salir."
        )

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Efecto espejo para que sea natural
            frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detect(gray)

            self._draw_overlay_top(frame, "Buscando rostro...")
            self._draw_overlay_bottom(frame, "Q: salir")

            if len(faces) == 0:
                cv2.rectangle(frame, (5,5), (frame_w-5, frame_h-5), (0,0,200), 2)
            else:
                for (x, y, w, h) in faces:
                    rostro = gray[y:y+h, x:x+w]

                    try:
                        # LBPH devuelve: label (quién es) y confidence (distancia/diferencia)
                        label, confidence = self.recognizer.predict(rostro)
                        
                        # --- DEBUG EN CONSOLA ---
                        # Esto te ayudará a saber qué valor poner en THRESHOLD
                        nombre_temp = self.label_dict.get(label, "Unknown")
                        print(f"Detectado: {nombre_temp} | Diferencia: {int(confidence)}") 
                        # ------------------------

                    except:
                        label, confidence = -1, 999

                    # --- LÓGICA DE DECISIÓN ---
                    # Si la diferencia es MENOR al límite, es la persona.
                    if confidence < THRESHOLD:
                        nombre = self.label_dict.get(label, "Desconocido")
                        color = (0, 255, 0) # Verde
                        text_top = f"Hola, {nombre}"
                    else:
                        nombre = "Desconocido"
                        color = (0, 0, 255) # Rojo
                        text_top = "No identificado"

                    # Dibujar rectángulo y texto
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    
                    # Texto del nombre sobre la cabeza
                    cv2.putText(frame, nombre, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # Info extra abajo
                    self._draw_overlay_bottom(frame, f"Dif: {int(confidence)} (Limite: {THRESHOLD})")
                    self._draw_overlay_top(frame, text_top)

            cv2.imshow("Reconocimiento Facial", frame)
            
            # Salir con Q o ESC
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    Recognizer().start()