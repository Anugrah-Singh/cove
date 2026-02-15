import os
import numpy as np
import cv2
import concurrent.futures
from ai_engine import AIEngine
from cluster_engine import ClusterEngine
from person_manager import PersonManager
import time
import gc # Garbage Collector

def process_single_image(ai, img_path):
    try:
        img = cv2.imread(img_path)
        if img is None: return None
        faces = ai.get_faces(img)
        if not faces: return None
        # Select largest face
        faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
        return (img_path, faces[0].embedding)
    except Exception:
        return None

def main():
    # Reduced workers to 4 to save VRAM (Stability > Speed)
    MAX_WORKERS = 4 
    BATCH_SIZE = 500 # Process 500 images, then clear memory
    
    ai = AIEngine()
    
    img_folder = "test_images"
    all_images = [os.path.join(img_folder, f) for f in os.listdir(img_folder) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"🚀 STABLE TURBO: Scanning {len(all_images)} images in batches...")

    embeddings = []
    valid_paths = []
    
    start_time = time.time()

    # --- BATCH LOOP ---
    for i in range(0, len(all_images), BATCH_SIZE):
        batch_paths = all_images[i : i + BATCH_SIZE]
        
        # Process this batch only
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_single_image, ai, path): path for path in batch_paths}
            
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result:
                    path, emb = result
                    embeddings.append(emb)
                    valid_paths.append(path)

        # Force Memory Cleanup after each batch
        gc.collect()
        
        # Progress Update
        elapsed = time.time() - start_time
        fps = len(valid_paths) / elapsed
        print(f"Batch {i//BATCH_SIZE + 1} Done | Total: {len(valid_paths)}/{len(all_images)} | Speed: {fps:.2f} fps", end='\r')

    total_time = time.time() - start_time
    print(f"\n✅ Finished in {total_time:.2f}s ({len(valid_paths) / total_time:.2f} fps)")

    # --- CLUSTERING ---
    print("Running Clustering Logic...")
    if not embeddings:
        print("No faces found!")
        return

    emb_matrix = np.array(embeddings)
    clusterer = ClusterEngine(min_cluster_size=3, min_samples=2)
    labels = clusterer.fit_predict(emb_matrix)

    pm = PersonManager()
    pm.save_people(labels, valid_paths)
    print("✅ Database Updated.")

if __name__ == "__main__":
    main()