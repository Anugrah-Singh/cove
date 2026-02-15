import faiss
import numpy as np


class ClusterEngine:
    def __init__(
        self,
        min_cluster_size: int = 3,
        neighbor_k: int = 64,
        similarity_threshold: float | None = None,
    ):
        """Faiss-powered graph clustering: deterministic, fast, and noise-aware."""
        self.min_cluster_size = max(1, min_cluster_size)
        self.neighbor_k = max(2, neighbor_k)
        self.similarity_threshold = similarity_threshold

    def _auto_threshold(self) -> float:
        base = 0.62
        bump = min(0.25, (self.min_cluster_size - 2) * 0.015)
        return min(0.95, base + bump)

    def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
        print(f"   [FAISS] Clustering {len(embeddings)} faces...")

        if not isinstance(embeddings, np.ndarray):
            embeddings = np.array(embeddings)

        if embeddings.size == 0:
            return np.array([], dtype=int)

        embeddings = embeddings.astype('float32')
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        embeddings = embeddings / norms

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        k = min(self.neighbor_k, len(embeddings))
        distances, neighbors = index.search(embeddings, k)
        threshold = self.similarity_threshold or self._auto_threshold()

        labels = np.full(len(embeddings), -1, dtype=int)
        visited = np.zeros(len(embeddings), dtype=bool)
        cluster_id = 0

        for root in range(len(embeddings)):
            if visited[root]:
                continue

            component = []
            stack = [root]

            while stack:
                current = stack.pop()
                if visited[current]:
                    continue
                visited[current] = True
                component.append(current)

                for neighbor_idx, sim in zip(neighbors[current], distances[current]):
                    neighbor_idx = int(neighbor_idx)
                    if neighbor_idx == current or neighbor_idx < 0:
                        continue
                    if visited[neighbor_idx]:
                        continue
                    if sim >= threshold:
                        stack.append(neighbor_idx)

            if len(component) >= self.min_cluster_size:
                labels[component] = cluster_id
                cluster_id += 1

        noise = np.count_nonzero(labels == -1)
        print(
            f"   -> Found {cluster_id} clusters, {noise} noise faces (threshold={threshold:.2f})."
        )

        return labels