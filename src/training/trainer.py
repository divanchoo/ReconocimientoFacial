import cv2
import os
import numpy as np
from glob import glob

from deepface import DeepFace
from src.detection.face_detector import detect_faces
from src.preprocess.align import crop_align_face
from src.models.embedding import get_embedding
from src.db.models import init_db, add_user


class EmbeddingTrainer:
    def __init__(self):
        self.dataset_path = "src/data/dataset"     # igual que antes
        print("📂 Dataset en:", self.dataset_path)

    # --------------------------------------------------------------------
    def load_images(self):
        """
        Recorre src/data/dataset/
        Retorna:
            - diccionario {usuario: [imgs]}
        """
        data = {}

        for folder in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder)

            if not os.path.isdir(folder_path):
                continue

            print(f"📁 Leyendo fotos de: {folder}")
            imgs = []

            for filename in os.listdir(folder_path):
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(folder_path, filename)
                    img = cv2.imread(img_path)

                    if img is None:
                        print(f"⚠ No se pudo cargar {img_path}")
                        continue

                    imgs.append(img)

            if len(imgs) > 0:
                data[folder] = imgs

        return data

    # --------------------------------------------------------------------
    def train(self):
        print("🔍 Cargando imágenes del dataset...")
        dataset = self.load_images()

        if len(dataset) == 0:
            print("⚠ No hay datos en el dataset. Capture primero.")
            return

        print(f"🖼 Usuarios encontrados: {list(dataset.keys())}")

        # Inicializar DB
        init_db()

        # Entrenar usuario por usuario
        for user, images in dataset.items():
            embeddings = []

            print(f"\n🧠 Procesando usuario: {user}")

            for img in images:
                # detectar rostro
                boxes = detect_faces(img)
                if not boxes:
                    print("⚠ No se detectó rostro en una foto. Saltando...")
                    continue

                # primera cara detectada
                face_rgb = crop_align_face(img, boxes[0])

                emb = get_embedding(face_rgb)
                if emb is not None:
                    embeddings.append(emb)
                else:
                    print("⚠ No se pudo generar embedding. Saltando foto...")

            if len(embeddings) == 0:
                print(f"❌ No se generaron embeddings válidos para {user}.")
                continue

            # embedding promedio por usuario
            avg_emb = np.mean(np.array(embeddings), axis=0)

            # guardar en DB
            add_user(user, avg_emb)
            print(f"✔ Usuario '{user}' guardado con {len(embeddings)} embeddings válidos.")

        print("\n🎉 ENTRENAMIENTO COMPLETADO")
        print("Los usuarios ya están almacenados en la base de datos (SQLite).")


if __name__ == "__main__":
    EmbeddingTrainer().train()
