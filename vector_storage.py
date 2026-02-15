import json
import logging
import os

import faiss
import numpy as np

from vision_config import CONFIG, get_logger

logger = get_logger(__name__)


class VectorStorage:
    def __init__(self, dimension: int = 512, index_path: str = CONFIG.faiss_index_path, vector_path: str = None):
        self.dimension = dimension
        self.index_path = index_path
        self.vector_path = vector_path
        self.paths = []
        self.vector_matrix = None
        self.path_to_index = {}

        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
            except Exception as exc:
                logger.warning("Failed to read index (%s): %s", self.index_path, exc)
                self.index = faiss.IndexFlatIP(self.dimension)
        else:
            self.index = faiss.IndexFlatIP(self.dimension)

        self._load_paths()
        self._load_vectors()
        self._reindex_paths()

    def _load_paths(self):
        paths_file = f"{self.index_path}.paths"
        if not os.path.exists(paths_file):
            return
        try:
            with open(paths_file, "r") as f:
                self.paths = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            try:
                import pickle

                with open(paths_file, "rb") as f:
                    self.paths = pickle.load(f)
                with open(paths_file, "w") as f:
                    json.dump(self.paths, f)
            except Exception as inner_exc:
                logger.warning("Unable to migrate paths file: %s", inner_exc)
                self.paths = []
        except Exception as exc:
            logger.warning("Unable to load paths (%s): %s", paths_file, exc)

    def _load_vectors(self):
        if not self.vector_path or not os.path.exists(self.vector_path):
            return
        try:
            data = np.load(self.vector_path)
            self.vector_matrix = np.asarray(data, dtype="float32")
            if len(self.paths) != len(self.vector_matrix):
                logger.warning("Vector count mismatch: %d paths vs %d vectors", len(self.paths), len(self.vector_matrix))
        except Exception as exc:
            logger.warning("Failed to load cached vectors (%s): %s", self.vector_path, exc)
            self.vector_matrix = None

    def _reindex_paths(self):
        self.path_to_index = {path: idx for idx, path in enumerate(self.paths)}

    def _save_paths(self):
        paths_file = f"{self.index_path}.paths"
        try:
            with open(paths_file, "w") as f:
                json.dump(self.paths, f)
        except Exception as exc:
            logger.warning("Failed to persist paths (%s): %s", paths_file, exc)

    def _append_vectors(self, vectors: np.ndarray):
        if vectors is None or vectors.size == 0 or not self.vector_path:
            return
        if self.vector_matrix is None or len(self.vector_matrix) == 0:
            self.vector_matrix = vectors.copy()
        else:
            self.vector_matrix = np.vstack((self.vector_matrix, vectors))

    def _save_vectors(self):
        if self.vector_path is None or self.vector_matrix is None:
            return
        try:
            np.save(self.vector_path, self.vector_matrix)
        except Exception as exc:
            logger.warning("Failed to save vectors (%s): %s", self.vector_path, exc)

    def add(self, vectors: np.ndarray, new_paths: list):
        if vectors.size == 0:
            return
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        self.paths.extend(new_paths)
        self._append_vectors(vectors)
        self._reindex_paths()

    def search(self, query_vector: np.ndarray, k: int = 20):
        query_vector = np.atleast_2d(np.array(query_vector, dtype="float32"))
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, k)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.paths):
                results.append({"path": self.paths[idx], "score": float(score)})
        return results

    def save(self):
        try:
            faiss.write_index(self.index, self.index_path)
        except Exception as exc:
            logger.warning("Failed to write index (%s): %s", self.index_path, exc)
        self._save_paths()
        self._save_vectors()

    def get_vector_by_path(self, path: str):
        idx = self.path_to_index.get(path)
        if idx is None or self.vector_matrix is None:
            return None
        return self.vector_matrix[idx]

    def has_path(self, path: str) -> bool:
        return path in self.path_to_index