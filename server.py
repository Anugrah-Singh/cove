# server.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from ai_engine import AIEngine
from search_engine import SearchEngine
from vector_storage import VectorStorage
import uvicorn
import cv2
import numpy as np
import os

app = FastAPI(title="VisionArchive Sidecar")

# Security: restrict file access to project directory
ALLOWED_ROOT = os.path.abspath(".")

# Global Singletons (Load once on startup)
ai_engine = None
search_engine = None
storage = None
search_storage = None

@app.on_event("startup")
async def startup_event():
    global ai_engine, search_engine, storage, search_storage
    # Initialize your engines here
    # This ensures models are loaded into VRAM only once
    ai_engine = AIEngine(model_name="buffalo_s")
    search_engine = SearchEngine()
    storage = VectorStorage()  # Face embeddings for indexing
    search_storage = VectorStorage(index_path="models/faiss_search_index.bin")  # CLIP search
    print("System Online: GPU Models Loaded")

class SearchQuery(BaseModel):
    text: str
    limit: int = 20

class ImageIndexRequest(BaseModel):
    file_path: str

@app.post("/search/text")
async def search_by_text(query: SearchQuery):
    # 1. Encode Text -> Vector
    vector = search_engine.get_text_embedding(query.text)
    # 2. Search FAISS
    results = search_storage.search(vector, k=query.limit)
    return {"results": results}

@app.post("/index/image")
async def index_image(request: ImageIndexRequest):
    file_path = os.path.abspath(request.file_path)
    
    # Security: restrict to project directory
    if not file_path.startswith(ALLOWED_ROOT):
        raise HTTPException(status_code=403, detail="Access denied: path outside allowed directory")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    # 1. Read Image
    img = cv2.imread(file_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")
        
    # 2. Extract Faces
    # Using the global AI engine instance
    faces = ai_engine.get_faces(img)
    
    if not faces:
        return {"status": "ignored", "reason": "no_faces_found"}
        
    # 3. Select Largest Face
    faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
    embedding = faces[0].embedding
    
    # 4. Add to Index
    # Reshape to (1, 512) and float32 for FAISS
    vector = np.array([embedding], dtype="float32")
    
    storage.add(vector, [file_path])
    storage.save() # Persist immediately for reliability
    
    return {"status": "indexed", "path": file_path}

if __name__ == "__main__":
    # Workers=1 is crucial because the AI models are not thread-safe across processes
    uvicorn.run(app, host="127.0.0.1", port=8000, workers=1)