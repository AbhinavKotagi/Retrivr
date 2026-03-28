"""
app.py
------
Retrievr — Sidebar Search Layout.

Layout
------
  SIDEBAR  — Search Images (text query → top-5 results)
             Caption Lookup (query → best image_id + caption)
             Reset App button

  MAIN     — Top:    Section 1 · Upload Images
                     Section 2 · Process Images
             Bottom: Section 3 · Search Results     (image grid)
                     Section 4 · Caption Lookup Result

State (st.session_state)
------------------------
  uploaded_images       list[dict]      Files saved this session
  processed_flag        bool            All uploads are indexed
  search_results        list|None       Last search output
  last_query            str             Query that produced search_results
  caption_lookup_result SearchResult|None
  caption_lookup_query  str
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from database import init_db, purge_all_data, purge_vector_data
from processor import save_uploaded_file, process_image, is_already_processed
from search import search_images, search_top_caption, index_exists

# ============================================================================
# Bootstrap
# ============================================================================

Path("storage/images").mkdir(parents=True, exist_ok=True)
Path("vectors").mkdir(parents=True, exist_ok=True)
init_db()

# ============================================================================
# Page config  — sidebar starts expanded so it is visible immediately
# ============================================================================

st.set_page_config(
    page_title="Retrievr",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# Session-state
# ============================================================================

_DEFAULTS: dict = {
    "uploaded_images":       [],
    "processed_flag":        False,
    "search_results":        None,
    "last_query":            "",
    "caption_lookup_result": None,
    "caption_lookup_query":  "",
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ============================================================================
# Helpers
# ============================================================================

COLS_PER_ROW = 3


def _render_image_safe(file_path: str, caption: str = "") -> None:
    try:
        st.image(Image.open(file_path), caption=caption, use_container_width=True)
    except Exception:
        st.warning("⚠️ Image file not found on disk.")


def _score_badge(score: float) -> str:
    pct = score * 100
    icon = "🟢" if pct >= 75 else ("🟡" if pct >= 50 else "🔴")
    return f"{icon} {pct:.1f}% match"


def _reset_app() -> None:
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()


# ============================================================================
# SIDEBAR — both search bars live here
# ============================================================================

with st.sidebar:
    st.title("🔍 Retrievr")
    st.caption("Semantic image search")
    st.divider()

    # ── Search Images ────────────────────────────────────────────────────────
    st.header("Search Images")
    st.caption("Find the top 5 most relevant images for your query.")

    sidebar_query = st.text_input(
        label="Image search query",
        placeholder='e.g. "a dog in the park"',
        label_visibility="collapsed",
        key="sidebar_search_input",
    )
    search_clicked = st.button(
        "🔎 Search", type="primary", use_container_width=True, key="sidebar_search_btn"
    )

    if search_clicked:
        if not sidebar_query.strip():
            st.warning("⚠️ Please enter a query.")
        elif not index_exists():
            st.warning("⚠️ No index yet — process images first.")
        else:
            with st.spinner("Searching…"):
                st.session_state.search_results = search_images(sidebar_query.strip(), k=5)
            st.session_state.last_query = sidebar_query.strip()

    st.divider()

    # ── Caption Lookup ───────────────────────────────────────────────────────
    st.header("Caption Lookup")
    st.caption("Returns the best-matching image ID and caption.")

    lookup_query = st.text_input(
        label="Caption lookup query",
        placeholder='e.g. "sunset over water"',
        label_visibility="collapsed",
        key="sidebar_lookup_input",
    )
    lookup_clicked = st.button(
        "🔍 Find", type="primary", use_container_width=True, key="sidebar_lookup_btn"
    )

    if lookup_clicked:
        if not lookup_query.strip():
            st.warning("⚠️ Please enter a query.")
        elif not index_exists():
            st.warning("⚠️ No index yet — process images first.")
        else:
            with st.spinner("Finding best match…"):
                st.session_state.caption_lookup_result = search_top_caption(lookup_query.strip())
            st.session_state.caption_lookup_query = lookup_query.strip()

    st.divider()

    # ── Session Reset ─────────────────────────────────────────────────────────
    st.caption("**Session**")
    if st.button("🔄 Reset Session", use_container_width=True,
                 help="Clears in-memory state only. DB and files untouched."):
        _reset_app()

    st.divider()

    # ── Developer Reset ───────────────────────────────────────────────────────
    st.caption("**🛠 Developer Tools**")

    confirm = st.checkbox(
        "I understand this is irreversible",
        key="dev_reset_confirm",
    )
    dev_reset_clicked = st.button(
        "🗑️ Wipe All Data",
        use_container_width=True,
        type="primary",
        key="dev_reset_btn",
        help="Deletes all DB records, image files, and FAISS index.",
        disabled=not confirm,
    )

    if dev_reset_clicked and confirm:
        # 1. Wipe all rows from the SQLite images table
        deleted_rows = purge_all_data()

        # 2. Delete all stored image files from disk
        images_dir = Path("storage/images")
        deleted_files = 0
        if images_dir.exists():
            for f in images_dir.iterdir():
                if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    f.unlink()
                    deleted_files += 1

        # 3. Wipe FAISS index + id_map contents in-place (files stay on disk)
        #    index_exists() keeps returning True, search works immediately
        #    (returns 0 results) without needing a restart.
        purge_vector_data()

        # 4. Clear session state and rerun
        for key in list(st.session_state.keys()):
            del st.session_state[key]

        st.success(
            f"✅ Wiped: {deleted_rows} DB record(s), "
            f"{deleted_files} image file(s), "
            "FAISS index and ID map cleared (files kept on disk)."
        )
        st.rerun()


# ============================================================================
# MAIN — header
# ============================================================================

st.title("🔍 Retrievr Prototype")
st.caption("Upload images · generate AI captions · search in plain English.")
st.divider()

# ============================================================================
# SECTION 1 — Upload Images
# ============================================================================

st.header("1 · Upload Images")

uploaded_files = st.file_uploader(
    label="Choose images (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="Select one or more images.",
)

if uploaded_files:
    already_saved: set[str] = {img["name"] for img in st.session_state.uploaded_images}

    for uf in uploaded_files:
        if uf.name not in already_saved:
            image_id, file_path = save_uploaded_file(uf)
            st.session_state.uploaded_images.append({
                "image_id": image_id,
                "file_path": str(file_path),
                "name": uf.name,
            })
            already_saved.add(uf.name)
            st.session_state.processed_flag = False

    total = len(st.session_state.uploaded_images)
    st.subheader(f"📁 {total} image(s) queued")

    for row_start in range(0, total, COLS_PER_ROW):
        row_items = st.session_state.uploaded_images[row_start : row_start + COLS_PER_ROW]
        cols = st.columns(COLS_PER_ROW)
        for col, img in zip(cols, row_items):
            with col:
                _render_image_safe(img["file_path"], caption=img["name"])

else:
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
    unprocessed = [
        img for img in st.session_state.uploaded_images
        if not is_already_processed(img["image_id"])
    ]

    if not unprocessed:
        st.session_state.processed_flag = True
        st.success(
            f"✅ All {len(st.session_state.uploaded_images)} image(s) processed "
            "and indexed. Use the **sidebar** to search."
        )
    else:
        st.write(
            f"**{len(unprocessed)}** of **{len(st.session_state.uploaded_images)}** "
            "image(s) need processing."
        )

        if st.button("⚙️ Process Images", type="primary"):
            progress_bar = st.progress(0.0, text="Starting…")
            log_area    = st.empty()
            log_lines: list[str] = []
            ok_count = 0

            with st.spinner("Processing… this may take a moment on first run."):
                for i, img in enumerate(unprocessed):
                    label = img["name"]
                    progress_bar.progress(
                        i / len(unprocessed),
                        text=f"Processing {label} ({i + 1}/{len(unprocessed)})…",
                    )
                    try:
                        caption = process_image(img["image_id"], Path(img["file_path"]))
                        log_lines.append(f"✅ **{label}** — *{caption}*")
                        ok_count += 1
                    except Exception as exc:
                        log_lines.append(f"❌ **{label}** — `{exc}`")
                    log_area.markdown("\n\n".join(log_lines))

            progress_bar.progress(1.0, text="Done!")
            if ok_count == len(unprocessed):
                st.session_state.processed_flag = True
            st.success(
                f"✅ Processed **{ok_count}** / **{len(unprocessed)}** image(s). "
                "You can now search using the **sidebar**."
            )

st.divider()

# ============================================================================
# SECTION 3 — Search Results  (bottom of main, driven by sidebar)
# ============================================================================

st.header("3 · Search Results")

if st.session_state.search_results is None:
    st.info("🔎 Use the **Search Images** bar in the sidebar to find images.")

elif not st.session_state.search_results:
    st.warning(
        f'No matches found for **"{st.session_state.last_query}"**. '
        "Try a different query or process more images."
    )

else:
    results = st.session_state.search_results
    st.subheader(f'Top {len(results)} result(s) for: *"{st.session_state.last_query}"*')

    for row_start in range(0, len(results), COLS_PER_ROW):
        row_items = results[row_start : row_start + COLS_PER_ROW]
        cols = st.columns(COLS_PER_ROW)
        for col, result in zip(cols, row_items):
            with col:
                _render_image_safe(result.file_path)
                st.markdown(
                    f"<div style='text-align:center;font-size:0.9rem;"
                    f"font-weight:600;margin:4px 0'>"
                    f"{_score_badge(result.score)}</div>",
                    unsafe_allow_html=True,
                )
                st.caption(f"📝 {result.caption or 'No caption available'}")

st.divider()

# ============================================================================
# SECTION 4 — Caption Lookup Result  (bottom of main, driven by sidebar)
# ============================================================================

st.header("4 · Caption Lookup Result")

if st.session_state.caption_lookup_result is None and not st.session_state.caption_lookup_query:
    st.info("🔍 Use the **Caption Lookup** bar in the sidebar to find the best matching caption.")

elif st.session_state.caption_lookup_result is None and st.session_state.caption_lookup_query:
    st.warning(
        f'No caption match found for **"{st.session_state.caption_lookup_query}"**.'
    )

else:
    hit = st.session_state.caption_lookup_result
    q   = st.session_state.caption_lookup_query

    st.subheader(f'Best match for: *"{q}"*')

    meta_col, img_col = st.columns([3, 2], vertical_alignment="top")

    with meta_col:
        st.markdown("**🆔 Image ID**")
        st.code(hit.image_id, language=None)

        st.markdown("**📝 Caption**")
        st.info(hit.caption or "No caption available")

        st.markdown("**📊 Similarity Score**")
        st.progress(
            min(hit.score, 1.0),
            text=f"{_score_badge(hit.score)}  ({hit.score * 100:.2f}%)",
        )

    with img_col:
        st.markdown("**🖼️ Matched Image**")
        _render_image_safe(hit.file_path)