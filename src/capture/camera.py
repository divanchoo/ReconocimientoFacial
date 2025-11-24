import cv2
import os
import time
from src.detection.face_detector import FaceDetector

# ---------- Config ----------
OVERLAY_ALPHA = 0.6
CONFIRMATION_TIME = 1.0  # segundos que muestra "Foto guardada"
MIN_FACE_RATIO_FOR_GOOD = 0.22  # relación cara/ancho frame para considerar 'buena'
MIN_FACE_RATIO_FOR_OK = 0.12
# ----------------------------

class CameraCapture:
    def __init__(self):
        self.face_detector = FaceDetector()

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

    def _quality_semantic(self, face_box, frame_w, frame_h):
        x, y, w, h = face_box

        face_ratio = w / float(frame_w)

        face_cx = x + w/2
        face_cy = y + h/2
        frame_cx = frame_w / 2
        frame_cy = frame_h / 2

        dist_norm = ((face_cx - frame_cx)**2 + (face_cy - frame_cy)**2)**0.5 / ((frame_w**2 + frame_h**2)**0.5)

        if face_ratio >= MIN_FACE_RATIO_FOR_GOOD and dist_norm < 0.15:
            return (0, 200, 0), "LISTO"   # verde

        if face_ratio >= MIN_FACE_RATIO_FOR_OK and dist_norm < 0.25:
            return (0, 200, 200), "OK"    # amarillo/cian

        return (0, 0, 200), "MALO"        # rojo


    def start_capture(self):
        name = input("Ingrese el nombre del usuario: ").strip()
        try:
            num_photos = int(input("¿Cuántas fotos desea capturar?: "))
        except:
            print("Número inválido. Uso 20 fotos por defecto.")
            num_photos = 20

        save_path = f"src/data/dataset/{name}"
        os.makedirs(save_path, exist_ok=True)

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ No se pudo abrir la cámara.")
            return

        print("📸 Presiona 'c' para capturar · 'q' para salir")

        count = 0
        last_confirmation_time = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error al leer la cámara.")
                break

            frame_h, frame_w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_detector.detect(gray)

            # Mensaje superior por defecto
            top_msg = "Presiona C para capturar"
            bottom_msg = "Q: salir"

            if len(faces) == 0:
                # No face
                self._draw_overlay_top(frame, top_msg)
                self._draw_overlay_bottom(frame, "No se detecta rostro · " + bottom_msg)
                # rojo borde global
                cv2.rectangle(frame, (5,5), (frame_w-5, frame_h-5), (0,0,200), 2)
            else:
                (x, y, w, h) = faces[0]
                color, quality_label = self._quality_semantic((x,y,w,h), frame_w, frame_h)
                # rectángulo sobre la cara
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                # etiqueta de calidad arriba
                self._draw_overlay_top(frame, f"Rostro detectado · {quality_label}")
                # bottom guidance + progreso
                self._draw_overlay_bottom(frame, f"Foto {count}/{num_photos} · Presiona C para guardar · Q: salir")

            # mostrar confirmación si la hubo
            if time.time() - last_confirmation_time < CONFIRMATION_TIME:
                # mensaje verde temporal en el centro inferior
                h, w = frame.shape[:2]
                cv2.putText(frame, "✔️ Foto guardada", (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,200,0), 3)

            cv2.imshow("Captura de Rostros", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("c") and len(faces) > 0:
                (x, y, w, h) = faces[0]
                rostro = gray[y:y+h, x:x+w]
                photo_path = f"{save_path}/{count}.jpg"
                cv2.imwrite(photo_path, rostro)
                print(f"✔️ Foto capturada: {photo_path}")
                count += 1
                last_confirmation_time = time.time()

                if count >= num_photos:
                    print("✅ Captura completada.")
                    break

            if key == ord("q"):
                print("🚪 Saliendo de la captura.")
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    CameraCapture().start_capture()
