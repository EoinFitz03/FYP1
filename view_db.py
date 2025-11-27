from db import Database

db = Database("system.db")

# Load all encodings
encodings, names = db.load_all_encodings()

print("=== USERS IN DATABASE ===")
shown = set()
for name in names:
    if name not in shown:
        print("-", name)
        shown.add(name)

print("\nTotal encodings stored:", len(encodings))
