import sys
from src.capture.camera import CameraCapture
from src.training.trainer import LBPHTrainer
from src.recognition.recognizer import Recognizer
from src.ui.dialogs import ModernUI

def main():
    # 1. Instanciamos las clases de lógica
    # (No ejecutamos nada aún, solo preparamos las herramientas)
    capturadora = CameraCapture()
    entrenador = LBPHTrainer()
    reconocedor = Recognizer()

    # 2. Definimos las funciones "puente"
    # Estas funciones conectan los botones de la UI con la lógica
    
    def ejecutar_captura(cantidad, nombre):  # <-- Agrega 'nombre' aquí
        try:
            # Ahora pasamos el nombre a tu clase de cámara
            # (Asegúrate de que tu CameraCapture.start_capture acepte el nombre)
            capturadora.start_capture(cantidad, nombre) 
        except Exception as e:
            print(f"Error: {e}")

    def ejecutar_entrenamiento():
        try:
            entrenador.train()
        except Exception as e:
            print(f"Error en entrenamiento: {e}")

    def ejecutar_reconocimiento():
        try:
            reconocedor.start()
        except Exception as e:
            print(f"Error en reconocimiento: {e}")

    # 3. Iniciamos la Interfaz Gráfica
    # Le pasamos nuestras funciones puente para que los botones sepan qué hacer
    app = ModernUI(
        on_capture=ejecutar_captura,
        on_train=ejecutar_entrenamiento,
        on_recognize=ejecutar_reconocimiento
    )

    # 4. Arrancamos el bucle principal de la ventana
    app.mainloop()

if __name__ == "__main__":
    main()