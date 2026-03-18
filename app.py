"""
app.py
------
Retrievr — Part 2: Image Upload + AI Processing Pipeline.

UI flow:
  Section 1 — Upload images, preview them.
  Section 2 — "Process Images" button triggers caption + embedding + storage.
"""

import streamlit as st
from pathlib import Path
from PIL import Image

from database import init_db
from processor import save_uploaded_file, process_image, is_already_processed

# ---------------------------------------------------------------------------
# One-time initialisation (runs on every cold start / page refresh)
# ---------------------------------------------------------------------------

# Ensure required directories exist
Path("storage/images").mkdir(parents=True, exist_ok=True)
Path("vectors").mkdir(parents=True, exist_ok=True)

# Ensure the SQLite database + table exist
init_db()

# ---------------------------------------------------------------------------
# Streamlit page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Retrievr", page_icon="🔍", layout="wide")
st.title("🔍 Retrievr Prototype")
st.caption("Upload images, generate captions, and build a searchable index.")
st.divider()

# ---------------------------------------------------------------------------
# Session-state keys
# ---------------------------------------------------------------------------
# processed_ids : set of image_ids that have been processed in this session,
#                 used to prevent duplicate processing on re-runs.

if "processed_ids" not in st.session_state:
    st.session_state.processed_ids: set[str] = set()

# Holds (image_id, Path) pairs for files saved this session but not yet
# processed — populated after upload, consumed on button click.
if "pending" not in st.session_state:
    st.session_state.pending: list[tuple[str, Path]] = []

# ---------------------------------------------------------------------------
# Section 1 — Upload
# ---------------------------------------------------------------------------

st.header("1 · Upload Images")

uploaded_files = st.file_uploader(
    label="Choose one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    help="JPG and PNG files are supported.",
)

if uploaded_files:
    # Save each file to disk (only once — guard against Streamlit re-runs)
    new_pending: list[tuple[str, Path]] = []

    for uf in uploaded_files:
        # Use the original filename as a lightweight de-dup key within the
        # current uploader state (Streamlit re-presents the same files on
        # every interaction until the user clears them).
        already_saved_names = {p.name for _, p in st.session_state.pending}
        if uf.name not in already_saved_names:
            image_id, file_path = save_uploaded_file(uf)
            new_pending.append((image_id, file_path))

    # Merge newly saved files into session state
    st.session_state.pending.extend(new_pending)

    # Preview grid — 4 columns
    st.subheader(f"Preview — {len(st.session_state.pending)} image(s) queued")
    cols = st.columns(min(4, len(st.session_state.pending)))

    for idx, (img_id, img_path) in enumerate(st.session_state.pending):
        with cols[idx % 4]:
            try:
                st.image(
                    Image.open(img_path),
                    caption=img_path.name,
                    use_container_width=True,
                )
            except Exception:
                st.warning(f"Could not preview {img_path.name}")

else:
    # Clear pending list when the uploader is empty (user removed files)
    st.session_state.pending = []
    st.info("Upload images above to get started.")

st.divider()

# ---------------------------------------------------------------------------
# Section 2 — Process
# ---------------------------------------------------------------------------

st.header("2 · Process Images")

if not st.session_state.pending:
    st.warning("⚠️ No images uploaded yet. Use Section 1 first.")
else:
    unprocessed = [
        (iid, fp)
        for iid, fp in st.session_state.pending
        if iid not in st.session_state.processed_ids
        and not is_already_processed(iid)   # also check DB for cross-session safety
    ]

    if not unprocessed:
        st.success("✅ All uploaded images have already been processed.")
    else:
        st.write(
            f"**{len(unprocessed)}** image(s) ready to process. "
            "Click the button to generate captions and build the search index."
        )

        if st.button("⚙️ Process Images", type="primary"):
            progress_bar = st.progress(0, text="Starting…")
            results_placeholder = st.empty()
            results: list[dict] = []

            for i, (image_id, file_path) in enumerate(unprocessed):
                progress_text = f"Processing {file_path.name} ({i + 1}/{len(unprocessed)})…"
                progress_bar.progress((i) / len(unprocessed), text=progress_text)

                try:
                    caption = process_image(image_id, file_path)
                    st.session_state.processed_ids.add(image_id)
                    results.append(
                        {"file": file_path.name, "caption": caption, "status": "✅"}
                    )
                except Exception as exc:
                    results.append(
                        {"file": file_path.name, "caption": str(exc), "status": "❌"}
                    )

            progress_bar.progress(1.0, text="Done!")

            # Show a results table
            st.subheader("Processing Results")
            for r in results:
                st.markdown(
                    f"{r['status']} **{r['file']}** — *{r['caption']}*"
                )

            st.success(
                f"✅ Processed {sum(1 for r in results if r['status'] == '✅')} "
                f"of {len(unprocessed)} image(s). "
                "Embeddings saved to `vectors/faiss.index`."
            )