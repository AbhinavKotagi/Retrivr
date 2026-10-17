import sqlite3

from retrievr.services.metadata import MetadataStore


def test_metadata_store_can_create_and_round_trip_image(tmp_path):
    store = MetadataStore(tmp_path / "retrievr-test.db")
    store.init_db()

    store.upsert_image(
        image_id="image-001",
        path="/tmp/example.jpg",
        caption="A small test image",
    )

    row = store.get_image_by_id("image-001")

    assert row is not None
    assert row["file_path"] == "/tmp/example.jpg"
    assert row["caption"] == "A small test image"
    assert row["indexed"] == 1
    assert row["created_at"]
    assert row["updated_at"]


def test_metadata_store_migrates_legacy_images_table(tmp_path):
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE images (
                image_id TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                caption TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO images (image_id, file_path, caption, created_at)
            VALUES (?, ?, ?, ?)
            """,
            ("legacy-001", "/tmp/legacy.jpg", "Legacy caption", "2025-01-01T00:00:00Z"),
        )

    store = MetadataStore(database_path)
    store.init_db()
    row = store.get_image_by_id("legacy-001")

    assert row is not None
    assert row["indexed"] == 0
    assert row["updated_at"] == "2025-01-01T00:00:00Z"
