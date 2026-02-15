import os
import numpy as np
import json
import time
from search_engine import SearchEngine
from vector_storage import VectorStorage
from vision_config import CONFIG

def main():
    print("🧠 SEMANTIC RE-INDEXING (Teaching AI to 'Read' your photos)...")
    
    # 1. Load the file list we already found
    if not os.path.exists(CONFIG.paths_file):
        print(f"❌ '{CONFIG.paths_file}' missing. Run production_pipeline.py first.")
        return
        
    with open(CONFIG.paths_file, "r") as f:
        paths = json.load(f)
        
    print(f"   -> Found {len(paths)} images to index.")
    
    # 2. Initialize CLIP (The Semantic Brain)
    searcher = SearchEngine()
    
    # 3. Build a SEPARATE FAISS index for CLIP semantic search
    #    This is distinct from faiss_index.bin which holds face embeddings.
    search_storage = VectorStorage(index_path=CONFIG.search_index_path, vector_path=CONFIG.vector_path)
    
    clip_vectors = []
    valid_paths = []
    
    start = time.time()
    
    for i, path in enumerate(paths):
        try:
            embedding = searcher.get_image_embedding(path)
            
            if embedding is not None:
                clip_vectors.append(embedding)
                valid_paths.append(path)
                
        except Exception as e:
            print(f"Skipping {path}: {e}")
            
        if i % 50 == 0:
            speed = (i + 1) / (time.time() - start)
            print(f"   Indexed {i}/{len(paths)} | Speed: {speed:.1f} img/sec", end='\r')

    print(f"\n✅ Re-indexing Complete! ({len(clip_vectors)} valid images)")
    
    # 4. Save to FAISS search index
    if clip_vectors:
        vectors = np.array(clip_vectors, dtype="float32")
        search_storage.add(vectors, valid_paths)
        search_storage.save()
        print(f"💾 Saved semantic search index to '{CONFIG.search_index_path}'.")
    
    print("🚀 Semantic Search is now ready. Restart Streamlit to test.")

if __name__ == "__main__":
    main()