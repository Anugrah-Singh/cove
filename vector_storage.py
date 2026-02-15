# vector_storage.py
import faiss
import numpy as np
import json
import os

class VectorStorage:
    def __init__(self, dimension=512, index_path="models/faiss_index.bin"):
        self.dimension = dimension
        self.index_path = index_path
        self.paths = []
        
        # Check if index exists, else create new
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            paths_file = index_path + ".paths"
            if os.path.exists(paths_file):
                try:
                    with open(paths_file, "r") as f:
                        self.paths = json.load(f)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    # Legacy pickle format - migrate to JSON
                    import pickle
                    with open(paths_file, "rb") as f:
                        self.paths = pickle.load(f)
                    # Re-save as JSON
                    with open(paths_file, "w") as f:
                        json.dump(self.paths, f)
                    print("[INFO] Migrated paths file from pickle to JSON format.")
            
            # Validate consistency
            if self.index.ntotal != len(self.paths):
                print(f"[WARNING] Index/paths mismatch: {self.index.ntotal} vectors vs {len(self.paths)} paths. Re-index recommended.")
        else:
            # "Flat" index is exact search (good for <1M images)
            # For >1M, use "IVF" (approximate)
            self.index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (Cosine sim if normalized)

    def add(self, vectors, new_paths):
        """
        vectors: np.ndarray of shape (N, 512), dtype=float32
        new_paths: list of strings
        """
        faiss.normalize_L2(vectors) # Crucial for Cosine Similarity
        self.index.add(vectors)
        self.paths.extend(new_paths)
        
    def search(self, query_vector, k=20):
        """
        query_vector: np.ndarray of shape (512,) or (1, 512)
        """
        query_vector = np.atleast_2d(np.array(query_vector, dtype="float32"))
        faiss.normalize_L2(query_vector)
        # D = distances (scores), I = indices
        D, I = self.index.search(query_vector, k)
        
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx != -1 and idx < len(self.paths):  # Bounds check
                results.append({"path": self.paths[idx], "score": float(score)})
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.index_path + ".paths", "w") as f:
            json.dump(self.paths, f)