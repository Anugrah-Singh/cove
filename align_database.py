import numpy as np
import json
import os

def fix_database():
    print("🔧 VISION ARCHIVE: Database Repair Tool")
    
    # 1. Look for vectors (Try both names)
    if os.path.exists("models/embeddings.npy"):
        print("   -> Found pipeline output (embeddings.npy)")
        vectors = np.load("models/embeddings.npy")
    elif os.path.exists("models/image_vectors.npy"):
        print("   -> Found existing app database (image_vectors.npy)")
        vectors = np.load("models/image_vectors.npy")
    else:
        print("❌ CRITICAL: No vector file found! You must run the pipeline first.")
        return

    # 2. Look for paths (The list of filenames)
    if os.path.exists("models/paths.json"):
        print("   -> Found pipeline output (paths.json)")
        with open("models/paths.json", "r") as f:
            paths = json.load(f)
    else:
        print("❌ CRITICAL: 'models/paths.json' is missing.")
        print("   The app cannot know which photo belongs to which vector.")
        return

    # 3. Validation
    if len(vectors) != len(paths):
        print(f"❌ DATA CORRUPTION: Found {len(vectors)} vectors but {len(paths)} file paths.")
        print("   You must re-run 'python test_pipeline_queue.py' to fix this.")
        return

    print(f"✅ Data Validated: {len(vectors)} images.")

    # 4. Convert List to Dictionary Map (Required for VectorStorage)
    # The App needs {"path/to/image.jpg": 0, "path/to/other.jpg": 1}
    data_map = {path: i for i, path in enumerate(paths)}

    # 5. Save in the Standard Format
    print("💾 Saving standardized database...")
    np.save("models/image_vectors.npy", vectors)
    with open("models/image_cache.json", "w") as f:
        json.dump(data_map, f)

    print("✅ DONE! You can now run 'streamlit run app.py'")

if __name__ == "__main__":
    fix_database()
    