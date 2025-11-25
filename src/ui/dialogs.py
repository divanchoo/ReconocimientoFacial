import customtkinter as ctk
from tkinter import messagebox

# Configuración visual global
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class ModernUI(ctk.CTk):
    # CORRECCIÓN: Los nombres de los parámetros ahora coinciden con app.py
    def __init__(self, on_capture, on_train, on_recognize):
        super().__init__()
        
        # Guardamos las funciones que vienen desde app.py
        self.logic_capturar = on_capture
        self.logic_entrenar = on_train
        self.logic_reconocer = on_recognize

        # Configuración de ventana
        self.title("Sistema de Reconocimiento Facial")
        self.geometry("400x480")
        self.resizable(False, False)

        # Marco principal (para márgenes bonitos)
        self.main_frame = ctk.CTkFrame(self, corner_radius=15)
        self.main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        # Título
        self.lbl_titulo = ctk.CTkLabel(
            self.main_frame, 
            text="Panel de Control", 
            font=("Roboto Medium", 24)
        )
        self.lbl_titulo.pack(pady=(30, 20))

        # --- BOTONES ---
        # Botón 1: Capturar
        self.crear_boton(
            "📷 Capturar Rostro", 
            self.evento_capturar, 
            color="#3B8ED0", 
            hover="#36719F"
        )
        
        # Botón 2: Entrenar
        self.crear_boton(
            "🧠 Entrenar Modelo", 
            self.evento_entrenar, 
            color="#E04F5F", 
            hover="#B03E4B"
        )
        
        # Botón 3: Reconocer
        self.crear_boton(
            "👁️ Reconocimiento", 
            self.evento_reconocer, 
            color="#2CC985", 
            hover="#229A65"
        )
        
        # Separador
        ctk.CTkFrame(self.main_frame, height=2, fg_color="gray30").pack(pady=20, padx=50, fill="x")

        # Botón Salir
        self.btn_salir = ctk.CTkButton(
            self.main_frame, 
            text="Salir", 
            command=self.quit,
            fg_color="transparent", 
            border_width=1, 
            text_color="gray70",
            hover_color="gray30"
        )
        self.btn_salir.pack(pady=(0, 20))

    def crear_boton(self, texto, comando, color, hover):
        btn = ctk.CTkButton(
            self.main_frame, 
            text=texto, 
            command=comando, 
            height=45, 
            font=("Roboto", 14),
            fg_color=color,
            hover_color=hover
        )
        btn.pack(pady=10, padx=40, fill="x")

    # --- EVENTOS INTERNOS ---
    
    def evento_capturar(self):
        # 1. Pedir CANTIDAD (Estilo Moderno)
        dialog_cant = ctk.CTkInputDialog(text="¿Cuántas fotos tomar? (1-300):", title="Paso 1: Cantidad")
        str_cant = dialog_cant.get_input()
        
        if str_cant and str_cant.isdigit():
            cant = int(str_cant)
            if 1 <= cant <= 300:
                # 2. Pedir NOMBRE (Estilo Moderno) - ¡NUEVO!
                dialog_nombre = ctk.CTkInputDialog(text="Escribe el nombre del usuario:", title="Paso 2: Datos")
                nombre = dialog_nombre.get_input()

                if nombre:
                    messagebox.showinfo("Listo", f"Iniciando captura para: {nombre}\n({cant} fotos)")
                    # Ahora pasamos AMBOS datos a la lógica (app.py)
                    self.logic_capturar(cant, nombre)
                else:
                    messagebox.showwarning("Cancelado", "Se necesita un nombre para continuar.")
            else:
                messagebox.showwarning("Error", "El número debe estar entre 1 y 300")
        elif str_cant:
             messagebox.showerror("Error", "Por favor ingresa un número válido")

    def evento_entrenar(self):
        messagebox.showinfo("Info", "Iniciando proceso de entrenamiento...")
        self.logic_entrenar()
        messagebox.showinfo("Éxito", "Modelo entrenado correctamente.")

    def evento_reconocer(self):
        messagebox.showinfo("Info", "Abriendo cámara para reconocimiento (Presiona 'q' para salir)...")
        self.logic_reconocer()