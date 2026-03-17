"""
database.py
-----------
Handles all SQLite database operations for Retrievr.
Database file: retrievr.db
Table: images
"""

import sqlite3
from pathlib import Path

# Path to the SQLite database file
DB_PATH = "retrievr.db"


def get_connection() -> sqlite3.Connection:
    """Create and return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    # Return rows as dict-like objects (accessible by column name)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """
    Initialize the database.
    Creates the 'images' table if it does not already exist.
    """
    conn = get_connection()
    try:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS images (
                image_id   TEXT PRIMARY KEY,
                file_path  TEXT NOT NULL,
                caption    TEXT,
                created_at TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def insert_image(image_id: str, path: str, caption: str) -> None:
    """
    Insert a new image record into the database.

    Args:
        image_id:  Unique identifier (UUID) for the image.
        path:      File path to the stored image.
        caption:   Text caption / description for the image.
    """
    from utils import get_timestamp  # local import to avoid circular deps

    conn = get_connection()
    try:
        conn.execute(
            """
            INSERT INTO images (image_id, file_path, caption, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (image_id, path, caption, get_timestamp()),
        )
        conn.commit()
    finally:
        conn.close()


def get_all_images() -> list[sqlite3.Row]:
    """
    Retrieve every image record from the database.

    Returns:
        A list of Row objects (accessible like dicts).
    """
    conn = get_connection()
    try:
        cursor = conn.execute("SELECT * FROM images ORDER BY created_at DESC")
        return cursor.fetchall()
    finally:
        conn.close()


def get_image_by_id(image_id: str) -> sqlite3.Row | None:
    """
    Retrieve a single image record by its ID.

    Args:
        image_id: The UUID of the image to look up.

    Returns:
        A Row object if found, otherwise None.
    """
    conn = get_connection()
    try:
        cursor = conn.execute(
            "SELECT * FROM images WHERE image_id = ?", (image_id,)
        )
        return cursor.fetchone()
    finally:
        conn.close()
