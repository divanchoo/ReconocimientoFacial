import os
import sys

# Importar módulos del proyecto
from src.capture.camera import CameraCapture
from src.training.trainer import LBPHTrainer
from src.recognition.recognizer import Recognizer


class FaceRecognitionUI:

    def clear_console(self):
        os.system("cls" if os.name == "nt" else "clear")

    def show_menu(self):
        self.clear_console()
        print("""
==============================
   Sistema de Reconocimiento
         Facial OpenCV
==============================

1. Capturar fotos para un usuario
2. Entrenar el modelo
3. Reconocer en tiempo real
4. Salir
""")

    def run(self):
        while True:
            self.show_menu()
            choice = input("Seleccione una opción: ")

            if choice == "1":
                print("\n Iniciando captura de imágenes...\n")
                CameraCapture().start_capture()
                input("\nPresione ENTER para continuar...")

            elif choice == "2":
                print("\n Entrenando el modelo LBPH...\n")
                LBPHTrainer().train()
                input("\nPresione ENTER para continuar...")

            elif choice == "3":
                print("\n Iniciando reconocimiento facial...\n")
                Recognizer().start()
                input("\nPresione ENTER para continuar...")

            elif choice == "4":
                print("\n Saliendo del sistema...")
                sys.exit()

            else:
                print("\n Opción inválida. Intente de nuevo.")
                input("\nPresione ENTER para continuar...")


if __name__ == "__main__":
    FaceRecognitionUI().run()
