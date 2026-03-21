"""
app.py
------
Retrievr — Final (Part 4): Demo-ready integration.

Sections
--------
  1 · Upload Images    — file uploader + thumbnail preview grid
  2 · Process Images   — BLIP caption + CLIP embed + FAISS index
  3 · Search Images    — natural-language query → ranked results
  4 · Results          — 3-per-row card grid with score badges

State machine (st.session_state)
---------------------------------
  uploaded_images : list[dict]   — {image_id, file_path, name} for every
                                   file saved to disk this session.
  processed_flag  : bool         — True once every pending image has been
                                   processed (or was already in the DB).
  search_results  : list | None  — last search output; None = no search yet.
  last_query      : str          — the query that produced search_results.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from database import init_db
from processor import save_uploaded_file, process_image, is_already_processed
from search import search_images, index_exists

# ============================================================================
# Bootstrap — runs on every Streamlit script execution
# ============================================================================

Path("storage/images").mkdir(parents=True, exist_ok=True)
Path("vectors").mkdir(parents=True, exist_ok=True)
init_db()

# ============================================================================
# Page config
# ============================================================================

st.set_page_config(
    page_title="Retrievr",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================================
# Session-state initialisation
# ============================================================================

def _init_state() -> None:
    """Ensure every session-state key exists with a safe default."""
    defaults: dict = {
        # List of dicts: {image_id: str, file_path: str, name: str}
        "uploaded_images": [],
        # True once all images in uploaded_images are in the DB + FAISS index
        "processed_flag": False,
        # Last search output (list[SearchResult]) or None
        "search_results": None,
        # The query string that produced search_results
        "last_query": "",
    }
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

_init_state()

# ============================================================================
# Helpers
# ============================================================================

COLS_PER_ROW = 3   # results grid width


def _render_image_safe(file_path: str, caption: str = "") -> None:
    """Render an image from disk; show a placeholder on any error."""
    try:
        st.image(Image.open(file_path), caption=caption, use_container_width=True)
    except Exception:
        st.warning("Image file not found.")


def _score_badge(score: float) -> str:
    """Return a colour-coded emoji label based on the cosine similarity."""
    pct = score * 100
    if pct >= 75:
        colour = "🟢"
    elif pct >= 50:
        colour = "🟡"
    else:
        colour = "🔴"
    return f"{colour} {pct:.1f}% match"


def _reset_app() -> None:
    """
    Clear all session state and rerun — returns the app to its initial state.
    Does NOT delete files from disk or the database (non-destructive reset).
    """
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ============================================================================
# Header
# ============================================================================

title_col, reset_col = st.columns([8, 1], vertical_alignment="bottom")

with title_col:
    st.title("🔍 Retrievr Prototype")
    st.caption(
        "Upload images · generate AI captions · search in plain English."
    )

with reset_col:
    if st.button("🔄 Reset App", help="Clear session and start over"):
        _reset_app()

st.divider()

# ============================================================================
# SECTION 1 — Upload Images
# ============================================================================

st.header("1 · Upload Images")

uploaded_files = st.file_uploader(
    label="Choose images (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Select one or more images. You can add more before processing.",
)

if uploaded_files:
    # ------------------------------------------------------------------
    # Save any newly seen files to disk (guard against Streamlit re-runs
    # replaying the same uploader object on every interaction).
    # ------------------------------------------------------------------
    already_saved_names: set[str] = {
        img["name"] for img in st.session_state.uploaded_images
    }

    for uf in uploaded_files:
        if uf.name not in already_saved_names:
            image_id, file_path = save_uploaded_file(uf)
            st.session_state.uploaded_images.append(
                {
                    "image_id": image_id,
                    "file_path": str(file_path),
                    "name": uf.name,
                }
            )
            already_saved_names.add(uf.name)
            # Any new upload invalidates the processed flag
            st.session_state.processed_flag = False

    # ------------------------------------------------------------------
    # Thumbnail preview grid — 3 per row
    # ------------------------------------------------------------------
    total = len(st.session_state.uploaded_images)
    st.subheader(f"📁 {total} image(s) queued")

    for row_start in range(0, total, COLS_PER_ROW):
        row_items = st.session_state.uploaded_images[row_start : row_start + COLS_PER_ROW]
        cols = st.columns(COLS_PER_ROW)
        for col, img in zip(cols, row_items):
            with col:
                _render_image_safe(img["file_path"], caption=img["name"])

else:
    # Uploader cleared — wipe the in-session list so Section 2 stays honest.
    # Files remain on disk; the DB is untouched.
    if st.session_state.uploaded_images:
        st.session_state.uploaded_images = []
        st.session_state.processed_flag = False

    st.info("⬆️ Upload images above to get started.")

st.divider()

# ============================================================================
# SECTION 2 — Process Images
# ============================================================================

st.header("2 · Process Images")

if not st.session_state.uploaded_images:
    st.warning("⚠️ No images uploaded yet. Complete **Section 1** first.")

else:
    # Determine which images still need processing
    unprocessed = [
        img for img in st.session_state.uploaded_images
        if not is_already_processed(img["image_id"])
    ]

    if not unprocessed:
        # Every image is already in the DB
        st.session_state.processed_flag = True
        st.success(
            f"✅ All {len(st.session_state.uploaded_images)} image(s) are "
            "processed and indexed. Proceed to **Section 3** to search."
        )

    else:
        st.write(
            f"**{len(unprocessed)}** of "
            f"**{len(st.session_state.uploaded_images)}** image(s) need "
            "processing. Click the button to generate captions and build "
            "the search index."
        )

        if st.button("⚙️ Process Images", type="primary"):

            progress_bar = st.progress(0.0, text="Starting…")
            log_area = st.empty()
            log_lines: list[str] = []
            ok_count = 0

            with st.spinner("Processing images… this may take a moment on first run."):
                for i, img in enumerate(unprocessed):
                    label = img["name"]
                    progress_bar.progress(
                        i / len(unprocessed),
                        text=f"Processing {label} ({i + 1}/{len(unprocessed)})…",
                    )

                    try:
                        caption = process_image(
                            img["image_id"], Path(img["file_path"])
                        )
                        log_lines.append(f"✅ **{label}** — *{caption}*")
                        ok_count += 1
                    except Exception as exc:
                        log_lines.append(f"❌ **{label}** — `{exc}`")

                    log_area.markdown("\n\n".join(log_lines))

            progress_bar.progress(1.0, text="Done!")

            # Mark session as fully processed only when every image succeeded
            if ok_count == len(unprocessed):
                st.session_state.processed_flag = True

            st.success(
                f"✅ Processed **{ok_count}** / **{len(unprocessed)}** image(s). "
                "Embeddings saved to `vectors/faiss.index`."
            )

st.divider()

# ============================================================================
# SECTION 3 — Search Images
# ============================================================================

st.header("3 · Search Images")

# Guard 1: nothing uploaded
if not st.session_state.uploaded_images:
    st.warning("⚠️ Upload images in **Section 1** first.")
    st.stop()

# Guard 2: uploaded but not processed
if not st.session_state.processed_flag and not index_exists():
    st.warning(
        "⚠️ Images have not been processed yet. "
        "Click **Process Images** in **Section 2** to build the search index."
    )
    st.stop()

st.caption("Describe what you are looking for in plain English.")

query_col, btn_col = st.columns([6, 1], vertical_alignment="bottom")

with query_col:
    query = st.text_input(
        label="Search query",
        placeholder='e.g.  "a dog playing in the park"  or  "road at night"',
        label_visibility="collapsed",
        key="search_query_input",
    )

with btn_col:
    search_clicked = st.button(
        "🔎 Search", type="primary", use_container_width=True
    )

if search_clicked:
    if not query.strip():
        st.warning("⚠️ Please enter a search query before clicking Search.")
        st.stop()

    with st.spinner("Searching…"):
        results = search_images(query.strip(), k=5)

    # Persist results in session state — survives Streamlit re-runs
    st.session_state.search_results = results
    st.session_state.last_query = query.strip()

st.divider()

# ============================================================================
# SECTION 4 — Results
# ============================================================================

st.header("4 · Results")

# No search has been run yet
if st.session_state.search_results is None:
    st.info("🔎 Enter a query in **Section 3** and click **Search** to see results here.")
    st.stop()

results = st.session_state.search_results

# Search ran but returned nothing
if not results:
    st.warning(
        f'No matches found for **"{st.session_state.last_query}"**. '
        "Try a different query or process more images."
    )
    st.stop()

# Results header
st.subheader(
    f'Top {len(results)} result(s) for: *"{st.session_state.last_query}"*'
)

# 3-per-row card grid
for row_start in range(0, len(results), COLS_PER_ROW):
    row_items = results[row_start : row_start + COLS_PER_ROW]
    cols = st.columns(COLS_PER_ROW)

    for col, result in zip(cols, row_items):
        with col:
            # Image
            _render_image_safe(result.file_path)

            # Score badge — centred, bold
            st.markdown(
                f"<div style='text-align:center; font-size:0.9rem; "
                f"font-weight:600; margin:4px 0;'>"
                f"{_score_badge(result.score)}</div>",
                unsafe_allow_html=True,
            )

            # Caption
            st.caption(f"📝 {result.caption or 'No caption available'}")