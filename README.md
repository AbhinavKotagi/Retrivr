# Retrievr

Retrievr is a local AI image retrieval app that lets you upload images, generate captions, store metadata, index semantic embeddings, and rediscover images with natural-language search.

The current implementation is a Streamlit prototype with a modular Python service layer. It combines:

- **BLIP** for automatic image captions.
- **CLIP text embeddings** for query and caption vectors.
- **FAISS** for fast similarity search.
- **SQLite** for persistent image metadata.
- **Streamlit** for the upload, processing, search, and developer-reset UI.

## What the app does today

1. Users upload JPG, JPEG, or PNG files in the Streamlit UI.
2. Retrievr saves each file under `storage/images/` with a UUID filename.
3. The processing pipeline generates a caption with BLIP.
4. The caption is embedded with CLIP into a normalized 512-dimensional vector.
5. SQLite stores image metadata: image ID, file path, caption, indexed status, and timestamps.
6. FAISS incrementally appends the new vector and persists the updated index to disk.
7. Users search from the sidebar with plain-English prompts and see the top matching images.
8. Users can also run a caption lookup to retrieve the single best matching image ID and caption.

## Project structure

```text
Retrivr/
├── src/
│   └── retrievr/
│       ├── app.py                 # Streamlit UI and user workflow
│       ├── config.py              # Environment-driven settings and paths
│       ├── logging_config.py      # Shared logging bootstrap
│       ├── database.py            # Compatibility wrapper for metadata service
│       ├── embedding.py           # Compatibility wrapper for embedding service
│       ├── processor.py           # Compatibility wrapper for processing service
│       ├── search.py              # Compatibility wrapper for search service
│       ├── utils.py               # Shared utility helpers
│       └── services/
│           ├── captioning.py      # BLIP caption generation
│           ├── embedding_service.py # CLIP text embedding generation
│           ├── image_processing.py # Upload + caption + embed + persist orchestration
│           ├── metadata.py        # SQLite metadata database service
│           ├── search_service.py  # Semantic search and caption lookup service
│           └── vector_index.py    # Incremental FAISS index persistence
├── storage/images/                # Uploaded image files
├── vectors/faiss.index            # Persisted FAISS index
├── vectors/id_map.json            # FAISS row position -> image_id mapping
├── retrievr.db                    # SQLite metadata database
└── requirements.txt               # Python dependencies
```

## Architecture overview

Retrievr is intentionally split into small service modules so each major responsibility has one owner:

- **Configuration management** lives in `src/retrievr/config.py`. Defaults work out of the box, and environment variables can override paths, model names, embedding dimensions, and log level.
- **Logging** is initialized by `src/retrievr/logging_config.py` and used throughout the services for model loading, indexing, and error diagnostics.
- **Metadata persistence** is handled by `src/retrievr/services/metadata.py`, which creates and migrates the SQLite `images` table.
- **Captioning** is isolated in `src/retrievr/services/captioning.py` and falls back to `"An image"` if model inference fails.
- **Embedding** is isolated in `src/retrievr/services/embedding_service.py`; CLIP models are lazy-loaded and cached for the process.
- **Vector indexing** is isolated in `src/retrievr/services/vector_index.py`; new embeddings are appended incrementally instead of rebuilding the entire FAISS index.
- **Image processing orchestration** lives in `src/retrievr/services/image_processing.py`.
- **Search orchestration** lives in `src/retrievr/services/search_service.py`, which joins FAISS results back to SQLite metadata.

The application code now lives under the `src/retrievr/` package. Compatibility wrappers (`database.py`, `embedding.py`, `processor.py`, and `search.py`) remain inside that package, while implementation details live under `src/retrievr/services/`.

## SQLite metadata database

The SQLite database file defaults to `retrievr.db`. The app creates an `images` table with this metadata:

| Column | Purpose |
| --- | --- |
| `image_id` | UUID primary key used across storage, SQLite, and FAISS mapping. |
| `file_path` | Local path to the stored image file. |
| `caption` | BLIP-generated image description. |
| `indexed` | Boolean-like integer showing whether the vector has been written to FAISS. |
| `created_at` | UTC ISO-8601 creation timestamp. |
| `updated_at` | UTC ISO-8601 last update timestamp. |

Existing databases from earlier prototypes are migrated in place by adding missing `indexed` and `updated_at` columns.

## Incremental FAISS indexing

Retrievr uses `IndexFlatIP` with normalized CLIP embeddings, so inner product behaves like cosine similarity.

When an image is processed:

1. The service loads the existing FAISS index or creates a new empty one.
2. It loads `vectors/id_map.json`.
3. If the image ID is not already present, it appends the new embedding.
4. It appends the image ID to the ID map at the same row position.
5. It writes both files back to disk.

This keeps indexing incremental and avoids rebuilding vectors for previously processed images.

## Configuration

Runtime configuration is environment-driven. All variables are optional.

| Variable | Default | Description |
| --- | --- | --- |
| `RETRIVR_DB_PATH` | `retrievr.db` | SQLite database path. |
| `RETRIVR_STORAGE_DIR` | `storage/images` | Uploaded image storage directory. |
| `RETRIVR_VECTORS_DIR` | `vectors` | Directory for FAISS files. |
| `RETRIVR_FAISS_INDEX_NAME` | `faiss.index` | FAISS index filename. |
| `RETRIVR_ID_MAP_NAME` | `id_map.json` | FAISS-to-image-ID map filename. |
| `RETRIVR_EMBEDDING_DIM` | `512` | Embedding dimension for CLIP base patch32. |
| `RETRIVR_CLIP_MODEL` | `openai/clip-vit-base-patch32` | Hugging Face CLIP text model. |
| `RETRIVR_BLIP_MODEL` | `Salesforce/blip-image-captioning-base` | Hugging Face BLIP captioning model. |
| `RETRIVR_LOG_LEVEL` | `INFO` | Python logging level. |

Example:

```bash
RETRIVR_DB_PATH=data/retrievr.db RETRIVR_LOG_LEVEL=DEBUG streamlit run src/retrievr/app.py
```

## Setup & Installation

### 1. Clone the repository & create virtual environment

```bash
git clone <your-repo-url>
cd Retrivr

# Create virtual environment
python -m venv venv

# Activate (Windows PowerShell):
venv\Scripts\activate

# Activate (Mac/Linux):
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

*Note: On the first run, Hugging Face will automatically download BLIP and CLIP model weights locally into `~/.cache/huggingface`.*

### 3. Run the App

```bash
# Windows PowerShell:
$env:PYTHONPATH="src"; streamlit run src/retrievr/app.py

# Windows CMD:
set PYTHONPATH=src && streamlit run src/retrievr/app.py

# Mac / Linux:
PYTHONPATH=src streamlit run src/retrievr/app.py
```

Open the local Streamlit URL shown in the terminal (usually `http://localhost:8501`).

---

## 📊 Benchmarking & Evaluation Module

Retrievr includes an independent evaluation module located in `test/` for quantitative IR benchmarking against datasets like **Flickr8k**.

It computes 16 standard IR and performance metrics, including:
- **Recall@K** (K=1, 5, 10, 20)
- **Precision@K** (K=1, 5, 10, 20)
- **mAP@10** (Mean Average Precision)
- **MRR** (Mean Reciprocal Rank)
- **nDCG@10** (Normalized Discounted Cumulative Gain)
- **Latency Stats** (Average, Median, P95, Indexing Time)

### Running Evaluation

Place the Flickr8k dataset in `storage/Flickr 8k` or `test/dataset/flickr8k` (or specify `--dataset`), then run:

```bash
# Quick test (20 images):
python test/evaluation.py --limit 20

# Medium test (100 images):
python test/evaluation.py --limit 100

# Full benchmark (~8,000 images):
python test/evaluation.py
```

Results will be automatically printed to stdout and saved to `test/results/evaluation_results.json`.

---

## How to use Retrievr

1. Upload one or more images in **Section 1 · Upload Images**.
2. Click **Process Images** in **Section 2 · Process Images**.
3. Wait for captions and indexes to be generated.
4. Use **Search Images** in the sidebar to find the top semantic matches.
5. Use **Caption Lookup** in the sidebar to retrieve the single best matching image ID and caption.
6. Use **Reset Session** to clear only Streamlit session state.
7. Use **Wipe All Data** in developer tools to delete SQLite rows, image files, and vector contents.

## Data and reset behavior

- Uploaded images are persisted in `storage/images/`.
- Metadata is persisted in SQLite at `retrievr.db` by default.
- Vector search state is persisted in `vectors/faiss.index` and `vectors/id_map.json`.
- **Reset Session** does not delete persisted files or database records.
- **Wipe All Data** removes database rows and image files, then resets the FAISS index and ID map in place.

## Current limitations

- The app indexes captions rather than raw image embeddings.
- The FAISS index is append-only for normal ingestion; deletion is currently handled through full developer reset.
- Processing is local and CPU-friendly by default, but first model downloads can be slow.
- Authentication, multi-user storage, and production deployment hardening are not yet implemented.

