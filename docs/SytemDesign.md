System Design Document
Project Name: Retrievr
Domain: GenAI · AIML · Mobile Application
Design Strategy: Two-Stage System Evolution (Prototype → Production)
1. System Design Overview

Retrievr is designed using a progressive system architecture:

Stage 1 (Prototype): Validate AI pipeline and data flow with minimal infrastructure

Stage 2 (Production): Harden the system with security, scalability, and maintainability

📌 Core principle: Make it work → Make it reliable → Make it secure

2. High-Level Architecture
Components

Mobile Client (Flutter)

Backend API (Python)

AI Processing Pipeline

Database (Metadata + Vectors)

Image Storage

3. Stage 1 – Prototype System Design
3.1 Objective

Validate image ingestion → processing → retrieval

Avoid premature optimization

Keep infrastructure minimal

3.2 Architecture (Prototype)
Flutter App
   ↓
FastAPI Backend
   ↓
AI Model Inference
   ↓
Metadata DB + Vector Index
   ↓
Local Image Storage

3.3 Tech Stack (Prototype)
Frontend

Flutter

Basic Material UI

HTTP package

Backend

Python

FastAPI

Uvicorn

AI / ML

Pre-trained Vision-Language Model (CLIP / BLIP)

Sentence Transformers

Cosine similarity

Database

SQLite (metadata)

FAISS (local vector index)

Storage

Local filesystem

3.4 Folder Structure (Prototype)
Backend
backend/
│── main.py
│── api/
│   ├── upload.py
│   ├── search.py
│── services/
│   ├── image_processor.py
│   ├── embedding.py
│── db/
│   ├── metadata.db
│── vectors/
│   ├── faiss.index
│── storage/
│   ├── images/

Frontend
frontend/
│── lib/
│   ├── screens/
│   │   ├── upload_screen.dart
│   │   ├── search_screen.dart
│   ├── widgets/
│   ├── services/
│   │   ├── api_service.dart
│── main.dart

3.5 Database Setup (Prototype)

Metadata Table (SQLite):

Field	Type
image_id	UUID
file_path	TEXT
caption	TEXT
created_at	TIMESTAMP

Vector Storage

FAISS index stored locally

Mapping maintained via image_id

3.6 Security (Prototype)

No authentication

Local-only access

Hardcoded API endpoints

Intended only for development & testing

📌 Security intentionally minimal at this stage

3.7 API Keys (Prototype)

Stored in .env file

Loaded via environment variables

Example:

MODEL_NAME=clip-vit-base

4. Stage 2 – Production System Design (Finished Project)
4.1 Objective

Make the system secure, scalable, and maintainable

Prepare for real users and cloud deployment

4.2 Architecture (Production)
Flutter App
   ↓ HTTPS + Auth
API Gateway
   ↓
FastAPI Backend
   ↓
Async Task Queue
   ↓
AI Inference Service
   ↓
Metadata DB + Vector DB
   ↓
Cloud Image Storage

4.3 Tech Stack (Production)
Frontend

Flutter

Material 3

State management (Provider / Riverpod)

Backend

Python

FastAPI

Celery / BackgroundTasks

Docker

AI / ML

CLIP / BLIP (Dockerized)

Optional fine-tuning

Batch inference

Database

PostgreSQL (metadata)

Vector DB (FAISS / Chroma / Pinecone)

Storage

Cloud Object Storage (S3-like)

4.4 Folder Structure (Production)
Backend
backend/
│── app/
│   ├── main.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   ├── api/
│   │   ├── auth.py
│   │   ├── images.py
│   │   ├── search.py
│   ├── services/
│   │   ├── inference.py
│   │   ├── vector_store.py
│   ├── models/
│   ├── db/
│   │   ├── session.py
│   │   ├── schemas.py
│   ├── workers/
│   │   ├── tasks.py
│── docker/
│── .env
│── Dockerfile

4.5 Database Setup (Production)
Metadata (PostgreSQL)
Field	Type
image_id	UUID (PK)
user_id	UUID (FK)
image_url	TEXT
caption	TEXT
embedding_id	TEXT
created_at	TIMESTAMP
Vector Database

One embedding per image

Indexed by image_id

Cosine similarity search

4.6 Security (Production – 2nd Phase Focus)

✔ JWT-based authentication
✔ Secure API endpoints
✔ HTTPS enforced
✔ Role-based access (future)
✔ Input validation
✔ Rate limiting

4.7 API Keys & Secrets (Production)

Storage Rules:

Never hardcode keys

Use environment variables

Secrets managed via cloud secret manager

Example Keys:

OPENAI_API_KEY=****
DB_PASSWORD=****
JWT_SECRET=****
S3_ACCESS_KEY=****


📌 Keys are injected at runtime and never exposed to the client.

5. API Design (Common for Both Stages)
Endpoint	Method	Purpose
/upload	POST	Upload image
/process	POST	Trigger AI processing
/search	POST	Semantic search
/images/{id}	GET	Retrieve image
6. Scalability Considerations (Production)

Horizontal scaling of backend

Async inference pipeline

Cached embeddings

CDN for image delivery

7. System Evolution Summary
Aspect	Prototype	Production
DB	SQLite	PostgreSQL
Vectors	Local FAISS	Vector DB
Storage	Local FS	Cloud
Security	None	JWT + HTTPS
Deployment	Local	Docker + Cloud
8. Conclusion

This two-stage system design allows Retrievr to:

Validate AI functionality early

Reduce development risk

Gradually introduce security and scalability

Match real-world product engineering practices
