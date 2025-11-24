import cv2
import os
import time

from src.detection.face_detector import detect_faces
from src.preprocess.align import crop_align_face

# ---------- Config ----------
OVERLAY_ALPHA = 0.6
CONFIRMATION_TIME = 1.0  # segundos
MIN_FACE_RATIO_FOR_GOOD = 0.22
MIN_FACE_RATIO_FOR_OK = 0.12
# ----------------------------


class CameraCapture:

    def _draw_overlay_top(self, frame, text):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)
        cv2.putText(frame, text, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)

    def _draw_overlay_bottom(self, frame, text):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h-40), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, OVERLAY_ALPHA, frame, 1 - OVERLAY_ALPHA, 0, frame)
        cv2.putText(frame, text, (10, h-12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2)

    def _quality_semantic(self, box, frame_w, frame_h):
        top, right, bottom, left = box
        w = right - left
        h = bottom - top

        face_ratio = w / float(frame_w)

        # centro del rostro
        face_cx = left + w/2
        face_cy = top + h/2

        frame_cx = frame_w / 2
        frame_cy = frame_h / 2

        dist_norm = ((face_cx - frame_cx)**2 + (face_cy - frame_cy)**2)**0.5 / \
                    ((frame_w**2 + frame_h**2)**0.5)

        if face_ratio >= MIN_FACE_RATIO_FOR_GOOD and dist_norm < 0.15:
            return (0, 200, 0), "LISTO"   # verde

        if face_ratio >= MIN_FACE_RATIO_FOR_OK and dist_norm < 0.25:
            return (0, 200, 200), "OK"    # cian

        return (0, 0, 200), "MALO"        # rojo

    # --------------------------------------------------------------------------

    def start_capture(self):
        name = input("Ingrese el nombre del usuario: ").strip()
        if name == "":
            print("⚠ Nombre inválido.")
            return

        try:
            num_photos = int(input("¿Cuántas fotos desea capturar?: "))
        except:
            print("⚠ Número inválido. Usando 20 fotos.")
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

            # Detección con DeepFace
            boxes = detect_faces(frame)

            # Mensajes iniciales
            top_msg = "Presiona C para capturar"
            bottom_msg = f"Foto {count}/{num_photos} · Q: salir"

            if not boxes:
                self._draw_overlay_top(frame, top_msg)
                self._draw_overlay_bottom(frame, "No se detecta rostro · " + bottom_msg)
                cv2.rectangle(frame, (5, 5), (frame_w - 5, frame_h - 5), (0, 0, 200), 2)
            else:
                box = boxes[0]
                top, right, bottom, left = box
                color, quality_label = self._quality_semantic(box, frame_w, frame_h)

                # Rectángulo del rostro
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Etiqueta arriba
                self._draw_overlay_top(frame, f"Rostro detectado · {quality_label}")

                # Etiqueta abajo
                self._draw_overlay_bottom(frame, bottom_msg)

            # Mostrar "Foto guardada"
            if time.time() - last_confirmation_time < CONFIRMATION_TIME:
                h, w = frame.shape[:2]
                cv2.putText(frame, "✔ Foto guardada", (10, h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 3)

            cv2.imshow("Captura de Rostros", frame)
            key = cv2.waitKey(1) & 0xFF

            # ----- Guardar foto -----
            if key == ord("c") and boxes:
                face_rgb = crop_align_face(frame, boxes[0])  # recorte + resize 160x160 RGB

                if face_rgb is not None:
                    photo_path = f"{save_path}/{count}.jpg"
                    cv2.imwrite(photo_path, cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR))
                    print(f"✔ Foto capturada: {photo_path}")

                    count += 1
                    last_confirmation_time = time.time()

                    if count >= num_photos:
                        print("✅ Captura completada.")
                        break

            # ----- Salir -----
            if key == ord("q"):
                print("🚪 Saliendo de la captura.")
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    CameraCapture().start_capture()
