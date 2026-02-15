import os
import cv2
import numpy as np
import json
import time
import threading
import queue
import concurrent.futures
import warnings
import onnxruntime
import insightface
from insightface.app import FaceAnalysis
from vector_storage import VectorStorage

# Suppress Warnings
warnings.filterwarnings("ignore")
os.environ["ORT_LOGGING_LEVEL"] = "3"  # Silence ONNX logs

# --- CONFIGURATION ---
IMG_FOLDER = "test_images"
MODELS_DIR = "models"
EMBEDDINGS_FILE = os.path.join(MODELS_DIR, "embeddings.npy")
PATHS_FILE = os.path.join(MODELS_DIR, "paths.json")

# --- HARDWARE AUTOSCALER ---
_available_providers = onnxruntime.get_available_providers()
_USE_GPU = 'CUDAExecutionProvider' in _available_providers
# GPU: 2 parallel instances (~600MB VRAM each). CPU: 1 instance.
NUM_AI_INSTANCES = 2 if _USE_GPU else 1
DET_SIZE = (320, 320) # Turbo resolution

class AIWorker:
    """A self-contained AI worker that holds its own model instance."""
    def __init__(self, instance_id):
        self.id = instance_id
        print(f"   ⚙️ Booting AI Worker #{instance_id}...", end="\r")
        
        # Detect available providers
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if _USE_GPU else ['CPUExecutionProvider']
        ctx_id = 0 if _USE_GPU else -1
        
        self.app = FaceAnalysis(
            name='buffalo_s', 
            root='.', 
            allowed_modules=['detection', 'recognition'], 
            providers=providers
        )
        self.app.prepare(ctx_id=ctx_id, det_size=DET_SIZE)
        
        # FORCE HEURISTIC (The "Instant Start" Fix) - GPU only
        if _USE_GPU:
            sess_opt = {
                'device_id': 0,
                'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'HEURISTIC',
                'do_copy_in_default_stream': '1',
            }
            
            for model in self.app.models.values():
                if hasattr(model, 'session'):
                    model.session.set_providers(['CUDAExecutionProvider'], [sess_opt])
                
    def process(self, img):
        try:
            return self.app.get(img)
        except Exception:
            return []

def scan_worker(worker_id, file_queue, result_list, lock):
    """The worker thread that grabs files and runs inference."""
    # 1. Initialize dedicated AI engine for this thread
    ai = AIWorker(worker_id)
    
    while True:
        try:
            # Non-blocking get
            path = file_queue.get_nowait()
        except queue.Empty:
            break # Done
            
        try:
            # 2. READ (CPU)
            img = cv2.imread(path)
            if img is None: continue

            # 3. INFERENCE (GPU)
            faces = ai.process(img)

            if faces:
                # Sort by size (largest first)
                faces.sort(key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                
                # 4. SAVE (Thread-Safe)
                with lock:
                    result_list.append((path, faces[0].embedding))

        except Exception as e:
            print(f"Error: {e}")
        finally:
            file_queue.task_done()

def main():
    print("🚀 VISION ARCHIVE: AUTO-SCALING PIPELINE")
    print("=" * 50)
    
    # 1. SETUP DATA
    if not os.path.exists(IMG_FOLDER):
        print(f"❌ '{IMG_FOLDER}' not found.")
        return

    existing_paths = set()
    if os.path.exists(PATHS_FILE):
        with open(PATHS_FILE, 'r') as f:
            existing_paths = set(json.load(f))

    all_files = [os.path.join(IMG_FOLDER, f) for f in os.listdir(IMG_FOLDER) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    new_files = [f for f in all_files if f not in existing_paths]
    
    if not new_files:
        print("✅ Database is up to date.")
        return

    print(f"📂 Found {len(new_files)} new images.")
    print(f"⚙️  Hardware Strategy: {NUM_AI_INSTANCES} Parallel GPU Workers")
    
    # 2. FILL QUEUE
    file_queue = queue.Queue()
    for f in new_files:
        file_queue.put(f)

    # 3. START WORKERS
    new_results = []
    lock = threading.Lock()
    threads = []
    
    start_time = time.time()
    
    print("🔥 Starting Engines... (This takes ~2s)")
    for i in range(NUM_AI_INSTANCES):
        t = threading.Thread(target=scan_worker, args=(i, file_queue, new_results, lock))
        t.start()
        threads.append(t)

    # 4. MONITOR PROGRESS
    total = len(new_files)
    while any(t.is_alive() for t in threads):
        # Calculate stats
        done = total - file_queue.qsize()
        elapsed = time.time() - start_time
        if elapsed > 0:
            fps = done / elapsed
            # Dynamic ETA
            eta = (total - done) / fps if fps > 0 else 0
            
            bar_len = 30
            filled = int(done / total * bar_len)
            bar = "█" * filled + "░" * (bar_len - filled)
            
            print(f"   [{bar}] {done}/{total} | Speed: {fps:.1f} FPS | ETA: {eta:.0f}s   ", end='\r')
        time.sleep(0.5)

    # 5. MERGE & SAVE
    print("\n💾 Saving Database to FAISS...")
    
    if new_results:
        storage = VectorStorage()
        batch_paths, batch_embs = zip(*new_results)
        
        # Convert to numpy float32 for FAISS
        vectors = np.array(batch_embs, dtype="float32")
        
        storage.add(vectors, list(batch_paths))
        storage.save()
            
    print(f"✅ DONE! Added {len(new_results) if new_results else 0} new images.")

if __name__ == "__main__":
    main()