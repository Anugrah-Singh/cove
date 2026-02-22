import cv2
import json
import numpy as np
import os
import tqdm
from ai_engine import AIEngine
from vision_config import CONFIG

def main():
    print("🔄 REGENERATING FACE EMBEDDINGS (This might take a while)...")
    
    # 1. Load Paths
    paths_file = CONFIG.paths_file
    if not os.path.exists(paths_file):
        print(f"❌ '{paths_file}' missing.")
        return
        
    with open(paths_file, "r") as f:
        paths = json.load(f)
        
    print(f"Found {len(paths)} images to process.")
    
    # 2. Init AI Engine
    try:
        # Increase Workers for higher GPU saturation
        # Python GIL limits CPU threads, but we can launch multiple engines if VRAM allows.
        # However, for simply 'regenerate_faces', we can use a simpler ThreadPool.
        engine = AIEngine()
    except Exception as e:
        print(f"❌ Failed to init AI Engine: {e}")
        return

    # 3. Process
    embeddings = []
    valid_paths = []
    
    # Process only first 50 to test if needed, but here we do all
    # To be safe, let's process batches to save intermediate results? No, keep it simple.
    
    print(f"   Starting processing of {len(paths)} images...")
    
    # Use ThreadPoolExecutor to feed the GPU faster
    import concurrent.futures
    
    def process_one(path):
        try:
            img = cv2.imread(path)
            if img is None: return None
            faces = engine.app.get(img)
            if not faces: return None
            faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            return (path, faces[0].embedding)
        except:
            return None

    # Determine workers dynamically based on hardware
    workers = CONFIG.effective_workers
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(process_one, p): p for p in paths}
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(paths), unit="img"):
            res = future.result()
            if res:
                valid_paths.append(res[0])
                embeddings.append(res[1])
    
    # 4. Save
    print(f"✅ Generated {len(embeddings)} face embeddings.")
    
    # Save Embeddings
    np.save(CONFIG.embeddings_file, np.array(embeddings, dtype="float32"))
    print(f"💾 Saved to {CONFIG.embeddings_file}")
    
    # Update paths.json to match valid faces only
    with open(paths_file, "w") as f:
        json.dump(valid_paths, f)
    print(f"💾 Updated {paths_file}")
    
    # Also update faiss_index.bin.paths
    faiss_paths = f"{CONFIG.faiss_index_path}.paths"
    with open(faiss_paths, "w") as f:
        json.dump(valid_paths, f)
        
    print("🚀 Done! Now run 'python tune_clustering.py' to cluster faces.")

if __name__ == "__main__":
    main()