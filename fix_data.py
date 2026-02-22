import json
import numpy as np
import os
from vision_config import CONFIG

def main():
    print("🚑 VISION ARCHIVE: Emergency Data Fix 🚑")
    
    # 1. Load the corrupted files
    print(f"Loading paths from: {CONFIG.faiss_index_path}.paths")
    paths_file = f"{CONFIG.faiss_index_path}.paths"
    
    if not os.path.exists(paths_file):
        print("❌ Paths file not found.")
        return

    with open(paths_file, "r") as f:
        paths = json.load(f)
    
    print(f"Loading vectors from: {CONFIG.embeddings_file}")
    if not os.path.exists(CONFIG.embeddings_file):
        print("❌ Vector file not found.")
        return
        
    vectors = np.load(CONFIG.embeddings_file)
    
    # 2. Diagnose
    n_paths = len(paths)
    n_vecs = len(vectors)
    print(f"📊 Stats: {n_paths} paths | {n_vecs} vectors")
    
    if n_paths == n_vecs:
        print("✅ Data is already correct. No fix needed.")
        return
        
    if n_paths == n_vecs * 2:
        print("⚠️  DETECTED EXACT DUPLICATION! (Paths are double the vectors)")
        print("✂️  Trimming the second half of the paths list...")
        
        # Keep only the first half
        fixed_paths = paths[:n_vecs]
        
        # 3. Save
        with open(paths_file, "w") as f:
            json.dump(fixed_paths, f)
            
        print(f"✅ FIXED. Saved {len(fixed_paths)} paths to disk.")
        print("👉 You can now run 'streamlit run app.py'")
    else:
        print("❌ Mismatch is not a simple duplication. Manual inspection required.")

if __name__ == "__main__":
    main()