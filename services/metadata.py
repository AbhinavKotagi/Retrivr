"""SQLite metadata database service."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from config import settings
from utils import get_timestamp


class MetadataStore:
    """Persistence boundary for image metadata stored in SQLite."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else settings.database_path

    def get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self) -> None:
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS images (
                    image_id   TEXT PRIMARY KEY,
                    file_path  TEXT NOT NULL,
                    caption    TEXT,
                    indexed    INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            self._ensure_column(conn, "indexed", "INTEGER NOT NULL DEFAULT 0")
            self._ensure_column(conn, "updated_at", "TEXT")
            conn.execute("UPDATE images SET updated_at = created_at WHERE updated_at IS NULL")

    def _ensure_column(self, conn: sqlite3.Connection, name: str, definition: str) -> None:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(images)")}
        if name not in columns:
            conn.execute(f"ALTER TABLE images ADD COLUMN {name} {definition}")

    def upsert_image(self, image_id: str, path: str, caption: str, indexed: bool = True) -> None:
        now = get_timestamp()
        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO images (image_id, file_path, caption, indexed, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(image_id) DO UPDATE SET
                    file_path = excluded.file_path,
                    caption = excluded.caption,
                    indexed = excluded.indexed,
                    updated_at = excluded.updated_at
                """,
                (image_id, path, caption, int(indexed), now, now),
            )

    def get_all_images(self) -> list[sqlite3.Row]:
        with self.get_connection() as conn:
            return conn.execute("SELECT * FROM images ORDER BY created_at DESC").fetchall()

    def get_image_by_id(self, image_id: str) -> sqlite3.Row | None:
        with self.get_connection() as conn:
            return conn.execute("SELECT * FROM images WHERE image_id = ?", (image_id,)).fetchone()

    def mark_indexed(self, image_id: str) -> None:
        with self.get_connection() as conn:
            conn.execute(
                "UPDATE images SET indexed = 1, updated_at = ? WHERE image_id = ?",
                (get_timestamp(), image_id),
            )

    def purge_all_data(self) -> int:
        with self.get_connection() as conn:
            cursor = conn.execute("DELETE FROM images")
            return cursor.rowcount


metadata_store = MetadataStore()
