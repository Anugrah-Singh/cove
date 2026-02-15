import os
import cv2
import numpy as np
from collections import defaultdict
from ai_engine import AIEngine
from cluster_engine import ClusterEngine
from person_manager import PersonManager

def main():
    # 1. Initialize Engines
    ai = AIEngine()
    clusterer = ClusterEngine(min_cluster_size=3, min_samples=2)
    
    # 2. Load images
    folder = "test_images"
    image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
    
    if not image_paths:
        print("Please put some photos in 'test_images' folder!")
        return

    print(f"Scanning {len(image_paths)} images...")
    
    all_embeddings = []
    valid_paths = []

    # 3. SCAN LOOP
    for path in image_paths:
        img = cv2.imread(path)
        if img is None: continue
            
        faces = ai.get_faces(img)
        
        if not faces:
            continue
            
        # Sort by size (Largest face first)
        faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        
        all_embeddings.append(faces[0].embedding)
        valid_paths.append(path)
        print(f"Encoded {path}")

    if not all_embeddings:
        print("No faces found!")
        return

    # 4. CLUSTER
    print("\nRunning Clustering Logic...")
    emb_matrix = np.array(all_embeddings)
    labels = clusterer.fit_predict(emb_matrix)
    
    # 5. RESULTS
    groups = defaultdict(list)
    for path, label in zip(valid_paths, labels):
        if label == -1:
            groups["Noise/Unknown"].append(path)
        else:
            groups[f"Person_{label}"].append(path)
    
    print("\n--- FOUND PEOPLE ---")
    for person, photos in sorted(groups.items()):
        print(f"\n{person} found in {len(photos)} photos:")
        for p in photos:
            print(f" - {p}")

    pm = PersonManager()
    pm.save_people(labels, valid_paths)

if __name__ == "__main__":
    main()
