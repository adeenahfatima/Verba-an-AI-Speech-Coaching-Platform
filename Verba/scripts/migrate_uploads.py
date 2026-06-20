# LEGACY SCRIPT — superseded by Alembic (see /migrations).
# Only useful if you have an old local SQLite verba.db missing newer columns.
# Not used in production (Postgres uses Alembic migrations instead).

import sqlite3

conn = sqlite3.connect('instance/verba.db')
c = conn.cursor()

columns = [
    ("pitch_std", "REAL"),
    ("pitch_mean", "REAL"),
    ("volume_mean", "REAL"),
    ("volume_std", "REAL"),
    ("noise_level", "REAL"),
    ("vocab_richness", "REAL"),
    ("advanced_vocab_count", "INTEGER"),
    ("sentence_var", "REAL"),
    ("score", "INTEGER")
]

for col, typ in columns:
    try:
        c.execute(f"ALTER TABLE upload ADD COLUMN {col} {typ}")
    except sqlite3.OperationalError:
        pass  # Column already exists

conn.commit()
conn.close()
print("Migration complete.") 