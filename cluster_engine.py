import faiss
import numpy as np

class ClusterEngine:
    def __init__(
        self,
        min_cluster_size: int = 3,
        threshold: float = 0.55,  # 0.55 is a good starting point for Buffalo_S
    ):
        """
        Greedy Clustering (First-Leader):
        - Fast (O(N) with FAISS).
        - Prevents "chaining" (merging different people via a blurry link).
        - Threshold: 0.50 = Loose (merges more), 0.65 = Strict (splits duplicates).
        """
        self.min_cluster_size = max(1, min_cluster_size)
        self.threshold = threshold

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        print(f"   [Clustering] Processing {len(embeddings)} faces (Greedy, Thresh={self.threshold})...")

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        if embeddings.size == 0:
            return np.array([], dtype=int)

        # 1. Normalize Vectors (Critical for Cosine Similarity)
        embeddings = embeddings.astype('float32')
        faiss.normalize_L2(embeddings)

        # 2. Initialize State
        labels = np.full(len(embeddings), -1, dtype=int)
        
        # Leaders index: Stores the "Centroid" face of every cluster found so far
        dim = embeddings.shape[1]
        leader_index = faiss.IndexFlatIP(dim)
        
        # 3. Greedy Loop
        cluster_count = 0
        
        for i, vector in enumerate(embeddings):
            vector = vector.reshape(1, -1)
            
            if leader_index.ntotal == 0:
                # First face is always a new cluster
                leader_index.add(vector)
                labels[i] = cluster_count
                cluster_count += 1
                continue
            
            # Find the single best match among existing leaders
            dists, idxs = leader_index.search(vector, 1)
            best_score = dists[0][0]
            best_cluster_id = idxs[0][0]
            
            if best_score >= self.threshold:
                # Match found! Assign to that cluster
                labels[i] = best_cluster_id
            else:
                # No match. Create new cluster.
                leader_index.add(vector)
                labels[i] = cluster_count
                cluster_count += 1

        # 4. Filter Noise (Min Cluster Size)
        unique, counts = np.unique(labels, return_counts=True)
        valid_clusters = unique[counts >= self.min_cluster_size]
        
        # Mark small clusters as noise (-1)
        final_labels = np.array([
            lbl if lbl in valid_clusters else -1 
            for lbl in labels
        ])

        noise_count = np.count_nonzero(final_labels == -1)
        valid_count = len(valid_clusters)
        
        print(f"   -> Found {valid_count} valid clusters, {noise_count} noise faces.")
        return final_labels