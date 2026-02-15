import logging
import os
from typing import Optional, Tuple

try:
    import onnxruntime
except ImportError:
    onnxruntime = None


def _parse_bool(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


def _parse_tuple(value: str, default: Tuple[int, int]) -> Tuple[int, int]:
    try:
        parts = [int(p.strip()) for p in value.split(",") if p.strip()]
        if len(parts) == 2:
            return parts[0], parts[1]
    except ValueError:
        pass
    return default


def _detect_cuda_available() -> bool:
    if onnxruntime is None:
        return False
    try:
        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False


class VisionConfig:
    def __init__(self):
        self.model_dir = os.getenv("VISION_MODEL_DIR", "models")
        self.log_level = os.getenv("VISION_LOG_LEVEL", "INFO").upper()
        self.log_dir = os.getenv("VISION_LOG_DIR", "logs")
        self.log_file = os.path.join(self.log_dir, os.getenv("VISION_LOG_FILE", "vision_archive.log"))
        self.api_key = os.getenv("VISION_API_KEY")
        self.det_size = _parse_tuple(os.getenv("VISION_DET_SIZE", "320,320"), (320, 320))
        self._use_gpu_override = os.getenv("VISION_USE_GPU", "auto")
        self._force_cpu = _parse_bool(os.getenv("VISION_FORCE_CPU", None), False)
        self.skip_model_load = _parse_bool(os.getenv("VISION_SKIP_MODEL_LOAD", None), False)
        self.ai_workers = int(os.getenv("VISION_AI_WORKERS", "")) if os.getenv("VISION_AI_WORKERS") else None
        self.vector_path = os.path.join(self.model_dir, os.getenv("VISION_IMAGE_VECTOR_PATH", "image_vectors.npy"))
        self.faiss_index_path = os.path.join(self.model_dir, os.getenv("VISION_FAISS_INDEX", "faiss_index.bin"))
        self.search_index_path = os.path.join(self.model_dir, os.getenv("VISION_FAISS_SEARCH_INDEX", "faiss_search_index.bin"))
        self.people_db_path = os.path.join(self.model_dir, os.getenv("VISION_PEOPLE_DB_PATH", "people_db.json"))
        self.paths_file = os.path.join(self.model_dir, os.getenv("VISION_PATHS_FILE", "paths.json"))
        self.embeddings_file = os.path.join(self.model_dir, os.getenv("VISION_EMBEDDINGS_FILE", "embeddings.npy"))
        self.image_cache = os.path.join(self.model_dir, os.getenv("VISION_IMAGE_CACHE", "image_cache.json"))

    @property
    def use_gpu(self) -> bool:
        if self._force_cpu:
            return False
        if self._use_gpu_override.lower() in {"1", "true", "yes", "on"}:
            return True
        if self._use_gpu_override.lower() in {"0", "false", "no", "off"}:
            return False
        return _detect_cuda_available()

    @property
    def providers(self):
        return ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.use_gpu else ["CPUExecutionProvider"]

    @property
    def ctx_id(self):
        return 0 if self.use_gpu else -1

    @property
    def session_options(self):
        opts = {
            "arena_extend_strategy": "kNextPowerOfTwo",
            "do_copy_in_default_stream": "1",
        }
        if self.use_gpu:
            opts.update({
                "device_id": 0,
                "gpu_mem_limit": 4 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "HEURISTIC",
            })
        return opts

    @property
    def effective_workers(self) -> int:
        if self.ai_workers is not None:
            return max(1, self.ai_workers)
        return 2 if self.use_gpu else 1


CONFIG = VisionConfig()


def setup_logging():
    if logging.getLogger().handlers:
        return

    level = CONFIG.log_level
    os.makedirs(CONFIG.log_dir, exist_ok=True)
    handlers = [logging.StreamHandler()]
    try:
        handlers.append(logging.FileHandler(CONFIG.log_file, encoding="utf-8"))
    except OSError:
        pass

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def get_logger(name: Optional[str] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, CONFIG.log_level, logging.INFO))
    return logger


