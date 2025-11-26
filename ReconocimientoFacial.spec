# -- mode: python --

import os
import tkinter
from PyInstaller.utils.hooks import collect_data_files

datas = []

# --- TCL/TK para Python 3.13 ---
python_dir = os.path.dirname(os.path.dirname(tkinter.__file__))
tcl_root = os.path.join(python_dir, "tcl")

if os.path.isdir(tcl_root):
    datas.append((tcl_root, "tcl"))

# --- incluir carpeta src completa ---
datas.append(("src", "src"))

# --- incluir carpeta data explícitamente ---
datas.append(("src/data", "src/data"))

# --- AGREGADO IMPORTANTE: incluir modelo y labels ---
datas.append(("src/data/model.xml", "src/data"))
datas.append(("src/data/labels.pickle", "src/data"))

# --- OpenCV y numpy ---
datas += collect_data_files("cv2")
datas += collect_data_files("numpy")
datas += collect_data_files("customtkinter")

block_cipher = None

a = Analysis(
    ['app.py'],             # ENTRYPOINT
    pathex=['.'],
    binaries=[],
    datas=datas,
    hiddenimports=[
        "customtkinter",
        "PIL",
        "PIL.Image",
        "PIL.ImageTk",
        "cv2",
        "numpy",
    ],
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    name='ReconocimientoFacial',
    debug=False,
    strip=False,
    upx=True,
    console=False,
)