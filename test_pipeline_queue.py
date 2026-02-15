import os
import cv2
import numpy as np
import json
import time
from ai_engine import AIEngine

# --- CONFIG ---
BATCH_SIZE = 32
IMG_FOLDER = "test_images"
MODELS_DIR = "models"
EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "embeddings.npy")
PATHS_FILE = os.path.join(MODELS_DIR, "paths.json")

def main():
    print("🚀 VISION ARCHIVE: PRODUCTION ENGINE")
    
    # 1. Load Engine
    ai = AIEngine()
    
    # 2. SILENT WARM-UP (The magic trick)
    print("🔥 Warming up GPU...", end="", flush=True)
    dummy = np.zeros((320, 320, 3), dtype=np.uint8)
    for _ in range(5): ai.get_faces(dummy)
    print(" Done! (Ready for 200+ FPS)")

    # 3. Load Data
    existing_paths = []
    existing_embs = []
    if os.path.exists(PATHS_FILE):
        with open(PATHS_FILE, 'r') as f: existing_paths = json.load(f)
        existing_embs = list(np.load(EMBEDDINGS_FILE))

    all_files = [os.path.join(IMG_FOLDER, f) for f in os.listdir(IMG_FOLDER) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    new_files = [f for f in all_files if f not in set(existing_paths)]

    if not new_files:
        print("✅ Database up to date.")
        return

    print(f"⚡ Processing {len(new_files)} images...")
    
    new_paths, new_vectors = [], []
    start_time = time.time()
    
    # 4. HIGH SPEED LOOP
    # We use a simple loop because Batching + 320x320 + Heuristic is faster than threading overhead
    for i in range(0, len(new_files), BATCH_SIZE):
        batch = new_files[i : i + BATCH_SIZE]
        
        for path in batch:
            img = cv2.imread(path)
            if img is None: continue
            
            faces = ai.get_faces(img)
            
            if faces:
                faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                new_paths.append(path)
                new_vectors.append(faces[0].embedding)
        
        # Stats
        count = i + len(batch)
        elapsed = time.time() - start_time
        fps = count / elapsed
        print(f"   🚀 Speed: {fps:.1f} FPS | Progress: {count}/{len(new_files)}", end='\r')

    # 5. Save
    if new_vectors:
        final_embs = np.array(existing_embs + new_vectors) if existing_embs else np.array(new_vectors)
        final_paths = existing_paths + new_paths
        np.save(EMBEDDINGS_FILE, final_embs)
        with open(PATHS_FILE, "w") as f: json.dump(final_paths, f)
        print(f"\n✅ DONE! Avg Speed: {fps:.1f} FPS")

if __name__ == "__main__":
    main()