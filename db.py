import sqlite3
import json
import numpy as np
from datetime import datetime


class Database:
    """
    Simple SQLite wrapper for:
    - Users
    - Face encodings
    - Event logging
    """

    def __init__(self, path="system.db"):
        self.path = path
        self.conn = sqlite3.connect(self.path)
        self._create_tables()

    # CREATE TABLES
    def _create_tables(self):
        cur = self.conn.cursor()

        # USERS TABLE
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TEXT NOT NULL
        )
        """)

        # FACE ENCODINGS TABLE
        cur.execute("""
        CREATE TABLE IF NOT EXISTS face_encodings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            encoding TEXT NOT NULL,         -- JSON string (128 values)
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
        """)

        # EVENTS TABLE (optional but great for logs)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            gesture TEXT,
            action TEXT,
            ts TEXT NOT NULL
        )
        """)

        self.conn.commit()

    # USERS
    def add_user(self, name, role="user"):
        """Add a user and return their new user_id."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO users (name, role, created_at) VALUES (?, ?, ?)",
            (name, role, datetime.utcnow().isoformat())
        )
        self.conn.commit()
        return cur.lastrowid

    def get_user_id(self, name):
        """Return user_id for a given name, or None if not found."""
        cur = self.conn.cursor()
        cur.execute("SELECT id FROM users WHERE name = ?", (name,))
        row = cur.fetchone()
        return row[0] if row else None

    # FACE ENCODINGS
    def add_face_encoding(self, user_id, encoding: np.ndarray):
        """
        Save a 128-d face encoding for a user.
        Encoding is stored as JSON so it's easy to load back.
        """
        encoding_json = json.dumps(encoding.tolist())

        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO face_encodings (user_id, encoding, created_at) VALUES (?, ?, ?)",
            (user_id, encoding_json, datetime.utcnow().isoformat())
        )
        self.conn.commit()

    def load_all_encodings(self):
        """
        Returns:
            known_encodings: list of numpy arrays
            known_names:     list of names in the same order
        """
        cur = self.conn.cursor()
        cur.execute("""
        SELECT users.name, face_encodings.encoding
        FROM face_encodings
        JOIN users ON face_encodings.user_id = users.id
        """)

        rows = cur.fetchall()
        names = []
        encodings = []

        for name, enc_json in rows:
            arr = np.array(json.loads(enc_json), dtype=np.float64)
            names.append(name)
            encodings.append(arr)

        return encodings, names


    # EVENTS LOGGING
    def add_event(self, user_name, gesture, action):
        """Log a recognized event."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO events (user_name, gesture, action, ts) VALUES (?, ?, ?, ?)",
            (user_name, gesture, action, datetime.utcnow().isoformat())
        )
        self.conn.commit()


    # CLEANUP
    def close(self):
        self.conn.close()
