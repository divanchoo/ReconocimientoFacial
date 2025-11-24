# src/db/models.py
import sqlite3
import pickle
import numpy as np
import os
from typing import List, Tuple

# Crear carpeta si no existe
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "embeddings.db")

def init_db(db_path=DB_PATH):
    # Asegurar carpeta
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        embedding BLOB NOT NULL
    )
    """)
    conn.commit()
    conn.close()

def add_user(name: str, embedding: np.ndarray, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    emb_blob = pickle.dumps(embedding, protocol=pickle.HIGHEST_PROTOCOL)
    cur.execute("INSERT INTO users (name, embedding) VALUES (?, ?)", (name, emb_blob))
    conn.commit()
    conn.close()

def get_all_embeddings(db_path=DB_PATH) -> List[Tuple[int, str, np.ndarray]]:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT id, name, embedding FROM users")
    rows = cur.fetchall()
    out = []
    for r in rows:
        emb = pickle.loads(r[2])
        out.append((r[0], r[1], emb))
    conn.close()
    return out
