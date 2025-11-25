import cv2
import os
import time

# ---------- Configuración ----------
OVERLAY_ALPHA = 0.6
CONFIRMATION_TIME = 0.2 # Tiempo entre fotos automáticas
MIN_FACE_RATIO_FOR_GOOD = 0.22
MIN_FACE_RATIO_FOR_OK = 0.12
# -----------------------------------

class CameraCapture:
    def __init__(self):
        # Usamos el detector nativo de OpenCV para evitar errores de importación
        self.face_classif = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # --- Funciones Visuales (Overlays) ---
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
        
        # Centro de la cara vs Centro del frame
        face_cx = x + w/2
        face_cy = y + h/2
        frame_cx = frame_w / 2
        frame_cy = frame_h / 2

        # Distancia normalizada
        dist_norm = ((face_cx - frame_cx)**2 + (face_cy - frame_cy)**2)**0.5 / ((frame_w**2 + frame_h**2)**0.5)

        if face_ratio >= MIN_FACE_RATIO_FOR_GOOD and dist_norm < 0.15:
            return (0, 255, 0), "EXCELENTE"
        if face_ratio >= MIN_FACE_RATIO_FOR_OK and dist_norm < 0.25:
            return (0, 255, 255), "OK"
        return (0, 0, 255), "ACERCATE MAS"

    # --- Función Principal ---
    def start_capture(self, limit, name):
        # 1. Configuración de carpetas (Usamos 'Data' para consistencia)
        data_path = "Data"
        person_path = os.path.join(data_path, name)
        
        if not os.path.exists(person_path):
            os.makedirs(person_path)
            print(f"Carpeta creada: {person_path}")

        # 2. Iniciar cámara
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        count = 0
        last_capture_time = 0

        print(f"Iniciando captura inteligente para: {name}")

        while True:
            ret, frame = cap.read()
            if not ret: break

            # Espejo para que se sienta natural
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aux_frame = frame.copy()
            
            # Detección
            faces = self.face_classif.detectMultiScale(gray, 1.3, 5)

            # Mensajes base
            bottom_msg = "Presiona 'Q' para cancelar"

            if len(faces) == 0:
                self._draw_overlay_top(frame, "Buscando rostro...")
                self._draw_overlay_bottom(frame, bottom_msg)
                # Marco rojo indicando que no hay nadie
                cv2.rectangle(frame, (20,20), (frame_w-20, frame_h-20), (0,0,255), 2)
            else:
                # Tomamos el primer rostro detectado
                (x, y, w, h) = faces[0]
                
                # Evaluamos calidad
                color, quality_label = self._quality_semantic((x,y,w,h), frame_w, frame_h)
                
                # Dibujamos rectángulo del color de la calidad
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Lógica de Captura Automática
                # Si la calidad es OK o EXCELENTE y ha pasado el tiempo de espera
                time_diff = time.time() - last_capture_time
                
                if (quality_label in ["OK", "EXCELENTE"]) and (time_diff > CONFIRMATION_TIME):
                    if count < limit:
                        # Guardar foto
                        rostro = aux_frame[y:y+h, x:x+w]
                        rostro = cv2.resize(rostro, (150, 150), interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(f"{person_path}/rostro_{count}.jpg", rostro)
                        
                        count += 1
                        last_capture_time = time.time()
                        
                        # Feedback visual de "Foto tomada"
                        cv2.circle(frame, (50, 50), 20, (0, 255, 0), -1)

                # Actualizar textos
                self._draw_overlay_top(frame, f"Calidad: {quality_label} · Progreso: {int((count/limit)*100)}%")
                self._draw_overlay_bottom(frame, f"Capturadas: {count}/{limit} · {bottom_msg}")

            cv2.imshow("Sistema de Captura Inteligente", frame)

            k = cv2.waitKey(1)
            if k == 27 or k == ord('q') or count >= limit:
                break

        cap.release()
        cv2.destroyAllWindows()