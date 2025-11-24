import cv2
import os
import numpy as np


class LBPHTrainer:
    def __init__(self):
        self.dataset_path = "src/data/dataset"
        self.model_path = "src/data/model.xml"
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()

    def load_images(self):
        """
        Carga las imágenes y genera:
        - lista de imágenes en escala de grises
        - lista de labels (enteros)
        - diccionario id -> nombre (para el reconocimiento)
        """

        face_images = []
        labels = []
        label_dict = {}
        current_label = 0

        # Recorrer carpetas en dataset
        for folder in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder)

            if not os.path.isdir(folder_path):
                continue

            print(f" Leyendo fotos de: {folder}")

            label_dict[current_label] = folder  # nombre del usuario

            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(folder_path, filename)

                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        print(f" No se pudo cargar {img_path}")
                        continue

                    face_images.append(img)
                    labels.append(current_label)

            current_label += 1

        return face_images, labels, label_dict

    def train(self):
        print("🔍 Cargando imágenes del dataset...")
        images, labels, label_dict = self.load_images()

        if len(images) == 0:
            print(" No hay imágenes en el dataset. Capture primero.")
            return

        print(f" Total imágenes: {len(images)}")
        print(" Entrenando modelo LBPH...")

        self.recognizer.train(images, np.array(labels))
        self.recognizer.write(self.model_path)

        print(f" Modelo entrenado y guardado en {self.model_path}")
        print(" Diccionario de usuarios:", label_dict)

        # Guardamos el diccionario en un archivo
        import pickle
        with open("src/data/labels.pickle", "wb") as f:
            pickle.dump(label_dict, f)

        print(" Diccionario de labels guardado en src/data/labels.pickle")


if __name__ == "__main__":
    LBPHTrainer().train()
