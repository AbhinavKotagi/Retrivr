"""Backward-compatible SQLite metadata helpers backed by MetadataStore."""

from __future__ import annotations

from services.metadata import metadata_store
from services.vector_index import vector_index_service

DB_PATH = str(metadata_store.db_path)


def get_connection():
    return metadata_store.get_connection()


def init_db() -> None:
    metadata_store.init_db()


def insert_image(image_id: str, path: str, caption: str) -> None:
    metadata_store.upsert_image(image_id, path, caption, indexed=True)


def get_all_images():
    return metadata_store.get_all_images()


def get_image_by_id(image_id: str):
    return metadata_store.get_image_by_id(image_id)


def purge_all_data() -> int:
    return metadata_store.purge_all_data()


def purge_vector_data() -> None:
    vector_index_service.reset()
