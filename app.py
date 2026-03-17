"""
app.py
------
Entry point for the Retrievr Streamlit application.
Part 1: Foundation Setup — initialises storage, database, and confirms readiness.
"""

import streamlit as st
from pathlib import Path

from database import init_db

# ---------------------------------------------------------------------------
# Folder initialisation
# Ensure required directories exist before anything else runs.
# ---------------------------------------------------------------------------

STORAGE_IMAGES_DIR = Path("storage/images")
VECTORS_DIR = Path("vectors")

STORAGE_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
VECTORS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Database initialisation
# Creates retrievr.db and the 'images' table if they don't already exist.
# ---------------------------------------------------------------------------

init_db()

# ---------------------------------------------------------------------------
# Streamlit UI — Part 1: basic scaffold only
# ---------------------------------------------------------------------------

st.title("Retrievr Prototype")
st.write("Setup Complete")
