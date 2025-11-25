# -*- mode: python ; coding: utf-8 -*-

import os
import tkinter
from PyInstaller.utils.hooks import collect_data_files

# --- RUTAS REALES DE TKINTER PARA PYTHON 3.13 ---
# tk e tcl están en:  .../Python313/tcl/
python_dir = os.path.dirname(os.path.dirname(tkinter.__file__))
tcl_root = os.path.join(python_dir, "tcl")

datas = []

# agregar tcl completo si existe
if os.path.isdir(tcl_root):
    datas.append((tcl_root, "tcl"))
else:
    print("⚠ ADVERTENCIA: NO encontré carpeta TCL en:", tcl_root)

# --- agregar carpeta src completa ---
datas.append(('src', 'src'))

# --- OpenCV y numpy ---
cv2_datas = collect_data_files('cv2')
numpy_datas = collect_data_files('numpy')

datas += cv2_datas + numpy_datas

block_cipher = None

a = Analysis(
    ['src/ui/app.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'cv2',
        'cv2.data',
        'cv2.face',
        'numpy',
        'tkinter',
        'tkinter.filedialog',
        'tkinter.messagebox',
        'tkinter.simpledialog'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
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
    console=False
)
