import json
import numpy as np
import os
from vision_config import CONFIG

def main():
    print("🚑 VISION ARCHIVE: Search Index Fixer 🚑")
    
    # Paths to files
    search_paths_file = f"{CONFIG.search_index_path}.paths"
    vector_file = CONFIG.vector_path
    
    # 1. Load Data
    if not os.path.exists(search_paths_file):
        print(f"❌ '{search_paths_file}' missing.")
        return
    
    with open(search_paths_file, "r") as f:
        paths = json.load(f)
        
    if not os.path.exists(vector_file):
        print(f"❌ '{vector_file}' missing.")
        return
        
    vectors = np.load(vector_file)
    
    print(f"Structure: {len(paths)} paths | {len(vectors)} vectors")
    
    # 2. Logic to detect correct size from canonical source
    canonical_paths_file = CONFIG.paths_file
    if os.path.exists(canonical_paths_file):
        with open(canonical_paths_file, 'r') as f:
            valid_list = json.load(f)
            expected_count = len(valid_list)
            print(f"Expected count based on {canonical_paths_file}: {expected_count}")
            
            if len(paths) > expected_count:
                print(f"⚠️  Fixing Paths: keeping first {expected_count}")
                paths = paths[:expected_count]
            
            if len(vectors) > expected_count:
                print(f"⚠️  Fixing Vectors: keeping first {expected_count}")
                vectors = vectors[:expected_count]
                
            # 3. Save
            with open(search_paths_file, "w") as f:
                json.dump(paths, f)
            np.save(vector_file, vectors)
            
            print("✅ Fixed! Now run 'streamlit run app.py'")
    else:
        print("Cannot find canonical paths file to verify count.")

if __name__ == "__main__":
    main()