import cv2
import os
import numpy as np
import pickle
from tkinter import Tk, messagebox


class LBPHTrainer:
    def __init__(self):
        self.dataset_path = "src/data/dataset"
        self.model_path = "src/data/model.xml"
        self.labels_path = "src/data/labels.pickle"
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

            label_dict[current_label] = folder  # nombre del usuario

            for filename in os.listdir(folder_path):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(folder_path, filename)

                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    face_images.append(img)
                    labels.append(current_label)

            current_label += 1

        return face_images, labels, label_dict

    def train(self):
        # Ventanita oculta para mostrar alertas
        root = Tk()
        root.withdraw()

        images, labels, label_dict = self.load_images()

        if len(images) == 0:
            messagebox.showwarning(
                "Dataset vacío",
                "No hay imágenes en el dataset.\nCapture fotos primero."
            )
            return

        # Entrenamiento LBPH
        self.recognizer.train(images, np.array(labels))
        self.recognizer.write(self.model_path)

        # Guardar diccionario
        with open(self.labels_path, "wb") as f:
            pickle.dump(label_dict, f)

        messagebox.showinfo(
            "Entrenamiento completado",
            f"Modelo guardado en:\n{self.model_path}\n\n"
            f"Diccionario guardado en:\n{self.labels_path}"
        )


if __name__ == "__main__":
    LBPHTrainer().train()
