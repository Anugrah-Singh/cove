import numpy as np
import json
import os
import time
from cluster_engine import ClusterEngine
from person_manager import PersonManager

def main():
    print("⚡ VISION ARCHIVE: INSTANT TUNER (FAISS) ⚡")
    
    # 1. Load the "frozen" data
    if not os.path.exists("models/embeddings.npy"):
        print("❌ No checkpoint found. Run production_pipeline.py first.")
        return

    print("Loading face vectors from disk...", end="")
    embeddings = np.load("models/embeddings.npy")
    
    if os.path.exists("models/paths.json"):
        with open("models/paths.json", "r") as f:
            paths = json.load(f)
    else:
        paths = []
        
    print(f" Done. ({len(embeddings)} faces)")

    # Default settings
    current_threshold = 0.55
    min_cluster_size = 3

    while True:
        print("\n" + "="*50)
        print(f"Current Settings: Threshold={current_threshold} | Min Size={min_cluster_size}")
        print("---")
        print("Higher Threshold (e.g., 0.65) = Stricter, splits people up (use if merging happens)")
        print("Lower Threshold  (e.g., 0.45) = Looser, merges lookalikes (use if splitting happens)")
        print("---")
        
        val = input("Enter new Threshold (0.1 - 0.9) or 'exit': ")
        if val.lower() == 'exit': break
        
        try:
            new_threshold = float(val)
            if not (0.0 < new_threshold < 1.0):
                print("⚠️  Please enter a number between 0.0 and 1.0")
                continue
            current_threshold = new_threshold
        except ValueError:
            print("⚠️  Invalid number.")
            continue

        start = time.time()
        
        # 2. Run Clustering instantly
        clusterer = ClusterEngine(threshold=current_threshold, min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(embeddings)

        # 3. Analyze Results
        unique_people = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = list(labels).count(-1)
        
        print(f"✅ Re-clustered in {time.time() - start:.2f}s")
        print(f"   -> Found {unique_people} unique people.")
        print(f"   -> {noise_count} faces marked as 'Unknown/Noise' (Cluster size < {min_cluster_size}).")

        # 4. Save?
        save = input("Save this result to Database? (y/n): ")
        if save.lower() == 'y':
            pm = PersonManager()
            pm.save_people(labels, paths, overwrite=True) 
            print("💾 Saved. Run 'streamlit run app.py' to view the new clusters.")

if __name__ == "__main__":
    main()