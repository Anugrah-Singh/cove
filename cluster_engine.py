import hdbscan
import numpy as np

class ClusterEngine:
    def __init__(self, min_cluster_size=3, min_samples=None):
        """
        Production-Grade Clustering using HDBSCAN.
        - min_cluster_size: Smallest grouping to consider a "Person".
        - metric='euclidean': Since ArcFace vectors are normalized, 
          Euclidean distance is mathematically equivalent to Cosine but MUCH faster in HDBSCAN.
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples 
        
        # Core HDBSCAN algorithm (The "Heavy Lifter")
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean', 
            cluster_selection_method='eom', # 'Leaf' works better for small consistent clusters
            prediction_data=True # Allows us to predict new points later without retraining!
        )

    def fit_predict(self, embeddings):
        print(f"   [HDBSCAN] Clustering {len(embeddings)} faces...")
        
        # Safety Check: HDBSCAN expects float64 or float32
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)
            
        # Run Clustering
        labels = self.clusterer.fit_predict(embeddings)
        
        # HDBSCAN returns -1 for noise, just like DBSCAN
        unique_counts = np.unique(labels, return_counts=True)
        print(f"   -> Found {len(unique_counts[0]) - 1} clusters.")
        
        return labels