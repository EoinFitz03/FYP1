import sqlite3
import json # converting encodings to and from JSON
import numpy as np # used because face encodings are NumPy arrays
from datetime import datetime
# Import SQLite, JSON, NumPy, and timestamp tools used for database storage

class Database:
    """
    Simple SQLite wrapper for:
    - Users
    - Face encodings
    - Event logging
    """

    def __init__(self, path="system.db"): # Open the SQLite database file and ensure the required tables exist
        self.path = path # Store the path to the SQLite database file
        # # Create one SQLite connection for this Database instance
        self.conn = sqlite3.connect(self.path)
        # Open (or create) the SQLite database file
        self._create_tables()
        # creates tables 

    # CREATE TABLES
    def _create_tables(self): # Create the users, face_encodings, and events tables if they are missing
        cur = self.conn.cursor() # Create a cursor to execute SQL statements

        # USERS TABLE
         # Create the users table to store one row per enrolled person
        cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            name TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TEXT NOT NULL
        )
        """)

        # FACE ENCODINGS TABLE
        # Stores face encodings (the 128 numbers) linked to a user_id
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
        # # Create the events table to log recognised users, gestures, and actions over time
        cur.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            gesture TEXT,
            action TEXT,
            ts TEXT NOT NULL
        )
        """)
        # saves changes 
        self.conn.commit()

    # USERS
    def add_user(self, name, role="user"):
        """Add a user and return their new user_id."""
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO users (name, role, created_at) VALUES (?, ?, ?)", 
            # add new roles to users, name ,role,craeted at 
            (name, role, datetime.utcnow().isoformat())
            # gets teh UTC for the time and date 
        )
        self.conn.commit()
        return cur.lastrowid # Save any schema changes to the database

    def get_user_id(self, name):
        """Return user_id for a given name, or None if not found."""
        cur = self.conn.cursor()
         # Find the user id for the given name
        cur.execute("SELECT id FROM users WHERE name = ?", (name,))
        # fetchone() returns:
        # - a tuple like (3,) if a row exists
        # - None if nothing matched
        row = cur.fetchone()
        return row[0] if row else None

    # FACE ENCODINGS
    def add_face_encoding(self, user_id, encoding: np.ndarray):
        """
        Save a 128-d face encoding for a user.
        Encoding is stored as JSON so it's easy to load back.
        """
        encoding_json = json.dumps(encoding.tolist())
        # datase does not understand numpy this turns it inot a python list 
         # numpy array -> Python list -> JSON string

        cur = self.conn.cursor()
        # created cursor to talk to the DB 
         # Insert new encoding row linked to user_id
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
        # Match each face encoding with the user row where users.id equals face_encodings.user_id

        rows = cur.fetchall()
        names = []
        encodings = []
        # Take this user’s face vector, 
        # convert it to text, and save it as a row in the face_encodings table.

        for name, enc_json in rows:
        # Convert the JSON string back into a Python list,
        # then convert that list into a numpy array of floats.
        # This gives us the original 128-D face encoding as a numpy array.
            arr = np.array(json.loads(enc_json), dtype=np.float64)
             # numpy array -> Python list -> JSON string
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
