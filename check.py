import os
import sys

# 1. Verificar ruta del archivo
ruta_esperada = os.path.join("src", "ui", "dialogs.py")
print(f"--- DIAGNÓSTICO ---")
print(f"Directorio actual: {os.getcwd()}")
print(f"Buscando archivo en: {ruta_esperada}")

if os.path.exists(ruta_esperada):
    print("✅ Archivo físico encontrado.")
    
    # 2. Leer contenido físico
    with open(ruta_esperada, "r", encoding="utf-8") as f:
        contenido = f.read()
        if "class ModernUI" in contenido:
            print("✅ El archivo contiene 'class ModernUI'.")
        else:
            print("❌ PELIGRO: El archivo existe pero NO TIENE la clase ModernUI.")
            print("--- Contenido actual (primeras 5 lineas) ---")
            print(contenido[:200])
            print("--------------------------------------------")
else:
    print("❌ El archivo dialogs.py NO existe en src/ui/. Revisa tus carpetas.")

# 3. Intentar importar como lo hace Python
print("\n--- INTENTO DE IMPORTACIÓN ---")
try:
    from src.ui import dialogs
    print(f"Python importó dialogs desde: {dialogs.__file__}")
    
    if hasattr(dialogs, 'ModernUI'):
        print("✅ ÉXITO: Python puede ver ModernUI. Ahora debería funcionar.")
    else:
        print("❌ ERROR CRÍTICO: Python carga el archivo, pero no ve la clase ModernUI.")
        print("Posible causa: No guardaste el archivo o hay un archivo compilado (.pyc) viejo.")
        print("Solución: Ve a la carpeta src/ui/ y borra la carpeta __pycache__ si existe.")

except ImportError as e:
    print(f"❌ Falló la importación: {e}")