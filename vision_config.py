import logging
import os
import platform
import sys
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


def get_user_data_dir(app_name: str = "VisionArchive") -> str:
    system = platform.system()
    if system == "Windows":
        base = os.getenv("APPDATA")
    elif system == "Darwin":
        base = os.path.expanduser("~/Library/Application Support")
    else:
        base = os.path.expanduser("~/.config")

    path = os.path.join(base or os.getcwd(), app_name)
    os.makedirs(path, exist_ok=True)
    return path


def _has_required_assets(models_path: str) -> bool:
    if not os.path.isdir(models_path):
        return False
    clip_model = os.path.join(models_path, "clip_image.onnx")
    face_dir = os.path.join(models_path, "buffalo_s")
    return os.path.isfile(clip_model) and os.path.isdir(face_dir)


def _resolve_assets_dir(package_dir: str) -> str:
    targets = []
    override = os.getenv("VISION_MODEL_DIR")
    if override:
        targets.append(override)
    targets.extend([
        os.path.join(os.getcwd(), "models"),
        os.path.join(package_dir, "models"),
        os.path.join(os.path.dirname(package_dir), "models"),
    ])
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        targets.append(os.path.join(meipass, "models"))

    for candidate in targets:
        if candidate and _has_required_assets(candidate):
            return os.path.abspath(candidate)

    default_base = getattr(sys, "_MEIPASS", package_dir)
    return os.path.join(default_base, "models")


def _migrate_asset_file(source_dir: str, target: str, filename: str) -> None:
    src = os.path.join(source_dir, filename)
    if os.path.exists(src) and not os.path.exists(target):
        os.makedirs(os.path.dirname(target), exist_ok=True)
        try:
            from shutil import copy2

            copy2(src, target)
        except Exception:
            pass


class VisionConfig:
    def __init__(self):
        package_dir = os.path.dirname(os.path.abspath(__file__))
        self.assets_dir = _resolve_assets_dir(package_dir)
        self.assets_base = os.path.dirname(self.assets_dir)
        self.model_dir = self.assets_dir
        self.user_data_dir = os.getenv("VISION_USER_DATA", get_user_data_dir())
        os.makedirs(self.user_data_dir, exist_ok=True)

        self.log_level = os.getenv("VISION_LOG_LEVEL", "INFO").upper()
        self.log_dir = os.path.join(self.user_data_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(self.log_dir, os.getenv("VISION_LOG_FILE", "vision_archive.log"))
        self.api_key = os.getenv("VISION_API_KEY")
        self.det_size = _parse_tuple(os.getenv("VISION_DET_SIZE", "320,320"), (320, 320))
        self._use_gpu_override = os.getenv("VISION_USE_GPU", "auto")
        self._force_cpu = _parse_bool(os.getenv("VISION_FORCE_CPU", None), False)
        self.skip_model_load = _parse_bool(os.getenv("VISION_SKIP_MODEL_LOAD", None), False)
        self.ai_workers = int(os.getenv("VISION_AI_WORKERS", "")) if os.getenv("VISION_AI_WORKERS") else None
        self.vector_path = os.path.join(self.user_data_dir, os.getenv("VISION_IMAGE_VECTOR_PATH", "image_vectors.npy"))
        self.faiss_index_path = os.path.join(self.user_data_dir, os.getenv("VISION_FAISS_INDEX", "faiss_index.bin"))
        self.search_index_path = os.path.join(self.user_data_dir, os.getenv("VISION_FAISS_SEARCH_INDEX", "faiss_search_index.bin"))
        self.people_db_path = os.path.join(self.user_data_dir, os.getenv("VISION_PEOPLE_DB_PATH", "people_db.json"))
        self.paths_file = os.path.join(self.user_data_dir, os.getenv("VISION_PATHS_FILE", "paths.json"))
        self.embeddings_file = os.path.join(self.user_data_dir, os.getenv("VISION_EMBEDDINGS_FILE", "embeddings.npy"))
        self.image_cache = os.path.join(self.user_data_dir, os.getenv("VISION_IMAGE_CACHE", "image_cache.json"))

        self._migrate_cache()

    def _migrate_cache(self) -> None:
        migrate_candidates = [
            ("image_vectors.npy", self.vector_path),
            ("embeddings.npy", self.embeddings_file),
            ("paths.json", self.paths_file),
            ("people_db.json", self.people_db_path),
        ]
        for filename, target in migrate_candidates:
            _migrate_asset_file(self.assets_dir, target, filename)

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


