import numpy as np
import json
import os
import time
from cluster_engine import ClusterEngine
from person_manager import PersonManager

def main():
    print("⚡ VISION ARCHIVE: INSTANT TUNER ⚡")
    
    # 1. Load the "frozen" data
    if not os.path.exists("models/embeddings.npy"):
        print("❌ No checkpoint found. Run production_pipeline.py first.")
        return

    print("Loading face vectors from disk...", end="")
    embeddings = np.load("models/embeddings.npy")
    
    with open("models/paths.json", "r") as f:
        paths = json.load(f)
    print(f" Done. ({len(embeddings)} faces)")

    while True:
        print("\n" + "="*40)
        print("Higher min_cluster_size = fewer, larger clusters (requires more evidence)")
        print("Lower min_cluster_size  = more, smaller clusters (more sensitive)")
        try:
            val = input("Enter min_cluster_size (2 - 20) or 'exit': ")
            if val.lower() == 'exit': break
            min_cluster_size = int(val)
            if min_cluster_size < 2:
                print("min_cluster_size must be at least 2")
                continue
        except ValueError:
            print("Please enter an integer like 5")
            continue

        start = time.time()
        
        # 2. Run Clustering instantly
        clusterer = ClusterEngine(min_cluster_size=min_cluster_size, min_samples=None)
        labels = clusterer.fit_predict(embeddings)

        # 3. Analyze Results
        unique_people = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = list(labels).count(-1)
        
        print(f"✅ Re-clustered in {time.time() - start:.2f}s")
        print(f"   -> Found {unique_people} unique people.")
        print(f"   -> {noise_count} faces marked as 'Unknown/Noise'.")

        # 4. Save?
        save = input("Save this result to Database? (y/n): ")
        if save.lower() == 'y':
            pm = PersonManager()
            pm.save_people(labels, paths, overwrite=True) 
            print("💾 Saved. Run 'python gallery.py' to view the new clusters.")

if __name__ == "__main__":
    main()
