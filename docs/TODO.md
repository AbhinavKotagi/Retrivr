# Retrievr Project Roadmap

This document outlines the step-by-step process to build **Retrievr**, an AI-powered image retrieval application. The steps are based on the **Prototype First** approach defined in the system design documents.

## Phase 1: Project Setup & Initialization

- [ ] **Initialize Version Control**
    - [ ] Initialize Git repository in the root folder (`git init`).
    - [ ] Create a `.gitignore` file (include `venv/`, `__pycache__/`, `.env`, `build/`, `.dart_tool/`, `.idea/`).
- [ ] **Create Folder Structure**
    - [ ] Create `backend/` directory for Python code.
    - [ ] Create `frontend/` directory for Flutter code.
    - [ ] Ensure `docs/` directory contains all design files.

## Phase 2: Backend Development (Prototype)

### 2.1 Environment Setup
- [ ] **Python Environment**
    - [ ] Navigate to `backend/`.
    - [ ] Create a virtual environment: `python -m venv venv`.
    - [ ] Activate the virtual environment.
    - [ ] Create `requirements.txt` with dependencies:
        - `fastapi`
        - `uvicorn`
        - `python-multipart`
        - `torch`
        - `transformers`
        - `pillow`
        - `sentence-transformers` (or `clip` library options)
        - `faiss-cpu`
        - `numpy`
    - [ ] Install dependencies: `pip install -r requirements.txt`.

### 2.2 Core Application Structure
- [ ] **Project Skeleton**
    - [ ] Create `main.py` (Entry point).
    - [ ] Create directories: `api/`, `services/`, `db/`, `vectors/`, `storage/images/`.
- [ ] **Basic Server**
    - [ ] Implement basic `FastAPI` app in `main.py`.
    - [ ] Add a health check endpoint (`GET /health`).
    - [ ] Run server (`uvicorn main:app --reload`) to verify.

### 2.3 AI Pipeline Implementation
- [ ] **Image Processing Service** (`services/image_processor.py`)
    - [ ] Implement function to load image from file path using PIL.
    - [ ] Initialize Image Captioning Model (e.g., BLIP or CLIP).
    - [ ] Create function `generate_caption(image)` to get text description.
- [ ] **Embedding Service** (`services/embedding.py`)
    - [ ] Initialize Sentence Transformer model (e.g., `all-MiniLM-L6-v2` or CLIP text model).
    - [ ] Create function `get_text_embedding(text)` to convert captions/queries to vectors.

### 2.4 Database & Storage
- [ ] **Metadata Database** (`db/metadata.py`)
    - [ ] Set up SQLite connection.
    - [ ] Create table `images` with columns: `id` (UUID), `file_path`, `caption`, `created_at`.
    - [ ] Implement functions to `insert_image_metadata` and `get_all_images`.
- [ ] **Vector Storage** (`vectors/faiss_index.py`)
    - [ ] Initialize FAISS index (FlatL2 or similar).
    - [ ] Implement `add_vector(id, vector)` to store embeddings.
    - [ ] Implement `search_vectors(query_vector, k=5)` to retrieve nearest neighbors.

### 2.5 API Endpoints
- [ ] **Upload Endpoint** (`api/upload.py`)
    - [ ] Create `POST /upload`.
    - [ ] Accept image file.
    - [ ] Save locally to `storage/images/`.
    - [ ] Call AI pipeline: Generate Caption -> Generate Embedding.
    - [ ] Save Metadata to SQLite.
    - [ ] Save Vector to FAISS.
    - [ ] Return success response with Image ID and Caption.
- [ ] **Search Endpoint** (`api/search.py`)
    - [ ] Create `POST /search`.
    - [ ] Accept text query.
    - [ ] Convert query to embedding using `services/embedding.py`.
    - [ ] Search FAISS index for top matches.
    - [ ] Retrieve metadata for matched IDs from SQLite.
    - [ ] Return list of results (Image path, Caption, Score).

## Phase 3: Frontend Development (Prototype)

### 3.1 Flutter Setup
- [ ] **Initialize Flutter App**
    - [ ] Run `flutter create retrievr_app` inside `frontend/`.
    - [ ] Verify setup by running on emulator/device (`flutter run`).
- [ ] **Dependencies**
    - [ ] Add packages to `pubspec.yaml`:
        - `http` (for API calls)
        - `image_picker` (for selecting photos)
        - `path_provider` (optional, for file handling)
        - `dio` (alternative to http, optional)

### 3.2 UI Implementation
- [ ] **Upload Screen**
    - [ ] Create `UploadScreen` widget.
    - [ ] Add "Pick Image" button (Gallery/Camera).
    - [ ] Display selected image preview.
    - [ ] Add "Upload" button to send image to Backend `POST /upload`.
    - [ ] Show loading spinner during processing.
    - [ ] Show success message/generated caption upon completion.
- [ ] **Search Screen**
    - [ ] Create `SearchScreen` widget.
    - [ ] Add TextField for user query (e.g., "Dog in park").
    - [ ] Add "Search" button.
    - [ ] Implement `GridView` to display results.
- [ ] **Result Display**
    - [ ] Fetch results from Backend `POST /search`.
    - [ ] Display image thumbnails and captions in list/grid.
    - [ ] (Optional) Click to view full image.

## Phase 4: Integration & Testing

- [ ] **End-to-End Test**
    - [ ] Start Backend server.
    - [ ] Start Flutter app.
    - [ ] Upload a clear image (e.g., a cat).
    - [ ] Wait for processing.
    - [ ] Search for "cat" or "animal".
    - [ ] Verify the uploaded image appears in results.
- [ ] **Refinement**
    - [ ] Handle errors (Network issues, server errors).
    - [ ] Optimize image sizing for display.

## Phase 5: Future Production Considerations (Post-Prototype)

- [ ] **Database Migration**
    - [ ] Migrate SQLite to PostgreSQL.
    - [ ] Migrate local FAISS to Vector DB (Pinecone/Chroma).
- [ ] **Cloud Storage**
    - [ ] Move local image storage to AWS S3 / Google Cloud Storage.
- [ ] **Security**
    - [ ] Implement API Keys or OAuth.
    - [ ] Add Input Validation.
- [ ] **Dockerization**
    - [ ] Create `Dockerfile` for backend.
    - [ ] Create `docker-compose.yml` for full stack.
