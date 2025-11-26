import cv2
import os
import numpy as np
import pickle
from tkinter import messagebox

class LBPHTrainer:
    def __init__(self):
        # CORRECCIÓN: Usamos rutas absolutas para evitar confusiones entre 'Data', 'data' y 'src/data'
        # dataset_path: Donde la cámara guarda las fotos (Carpeta "Data" en la raíz del proyecto)
        self.dataset_path = os.path.abspath("Data")
        
        # output_path: Donde se guarda el modelo xml (Carpeta "src/data")
        self.output_path = os.path.abspath(os.path.join("src", "data"))
        
        # Aseguramos que exista la carpeta de salida
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

    def train(self):
        print(f"--- INICIANDO ENTRENAMIENTO ---")
        print(f"Buscando fotos en: {self.dataset_path}")
        print(f"Guardando modelo en: {self.output_path}")

        # Verificar si hay datos para entrenar
        if not os.path.exists(self.dataset_path):
            messagebox.showerror("Error", f"No existe la carpeta de datos:\n{self.dataset_path}\nCaptura fotos primero.")
            return

        people_list = os.listdir(self.dataset_path)
        # IMPORTANTE: Ordenar lista para garantizar que los IDs sean consistentes siempre
        people_list.sort()

        if len(people_list) == 0:
            messagebox.showwarning("Aviso", "La carpeta 'Data' está vacía. No hay usuarios para entrenar.")
            return

        print(f"Carpetas encontradas: {people_list}")

        faces = []
        labels = []
        label_dict = {} # Diccionario para guardar Nombre -> ID
        current_id = 0

        # Recorremos cada carpeta (cada persona)
        for person_name in people_list:
            person_path = os.path.join(self.dataset_path, person_name)
            
            # Solo procesar si es una carpeta
            if not os.path.isdir(person_path):
                continue

            # Asignamos un ID numérico a este nombre
            label_dict[current_id] = person_name 
            
            print(f"--> Procesando ID {current_id}: {person_name}")

            # Leemos cada foto dentro de la carpeta de la persona
            image_files = os.listdir(person_path)
            if not image_files:
                print(f"    AVISO: La carpeta {person_name} está vacía.")
                continue

            count_fotos = 0
            for image_name in image_files:
                # Ignorar archivos que no sean imágenes (como .DS_Store o Thumbs.db)
                if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue

                image_path = os.path.join(person_path, image_name)
                
                # Leer imagen en escala de grises
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None:
                    print(f"    Error leyendo imagen: {image_name}")
                    continue

                # Agregamos la imagen y su ID a las listas de entrenamiento
                faces.append(image)
                labels.append(current_id)
                count_fotos += 1
            
            print(f"    {count_fotos} fotos añadidas.")
            current_id += 1

        if len(faces) == 0:
            messagebox.showerror("Error", "No se encontraron rostros válidos para entrenar.")
            return

        # --- ENTRENAMIENTO LBPH ---
        print("Entrenando algoritmo LBPH...")
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(labels))

        # --- GUARDAR RESULTADOS ---
        
        # 1. Guardar el Modelo (.xml)
        recognizer.write(os.path.join(self.output_path, "model.xml"))
        
        # 2. Guardar las Etiquetas (.pickle)
        # Esto es lo CRUCIAL: Guardamos qué nombre corresponde a qué ID
        with open(os.path.join(self.output_path, "labels.pickle"), "wb") as f:
            pickle.dump(label_dict, f)

        print("-" * 30)
        print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print(f"Diccionario guardado: {label_dict}")
        print("-" * 30)
        # (Opcional) Mostrar mensaje en GUI, pero la GUI ya lo hace al terminar esta función