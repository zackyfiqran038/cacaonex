"""Simple local history; location and diagnosis are only saved when explicitly requested."""
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

DB = Path(__file__).parent / "data" / "history.db"

def connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB))
    conn.execute('''CREATE TABLE IF NOT EXISTS records (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL, diagnosis TEXT NOT NULL,
      confidence REAL NOT NULL, latitude REAL, longitude REAL,
      notes TEXT)''')
    conn.commit()
    return conn

def save(diagnosis, confidence, latitude=None, longitude=None, notes=''):
    with connect() as conn:
        conn.execute('''INSERT INTO records
            (created_at, diagnosis, confidence, latitude, longitude, notes)
            VALUES (?, ?, ?, ?, ?, ?)''',
            (datetime.now(timezone.utc).isoformat(timespec='seconds'),
             diagnosis, float(confidence), latitude, longitude, notes))

def all_records():
    with connect() as conn:
        return conn.execute('''SELECT id,created_at,diagnosis,confidence,latitude,longitude,notes
                            FROM records ORDER BY id DESC''').fetchall()

def delete_all():
    with connect() as conn:
        conn.execute('DELETE FROM records')
