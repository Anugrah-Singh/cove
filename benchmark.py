import time
import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

def benchmark():
    print("🕵️ VISION ARCHIVE: System Diagnostic Tool (Offline Mode)")
    print("=" * 50)
    
    # 1. VERIFY FILES PHYSICALLY BEFORE LOADING AI
    model_path = "models/buffalo_s"
    required_files = ["det_500m.onnx", "w600k_mbf.onnx"]
    
    missing = [f for f in required_files if not os.path.exists(os.path.join(model_path, f))]
    if missing:
        print(f"❌ ERROR: Missing local files in {model_path}: {missing}")
        print("Please ensure your models are in the correct subfolder.")
        return
    else:
        print(f"✅ Local models found in {model_path}. Proceeding offline...")

    img_dir = "test_images"
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) 
             if f.lower().endswith(('.jpg', '.png'))][:50]
    
    # --- TEST 1: DISK I/O SPEED ---
    print("\n1️⃣  TESTING DISK READ SPEED...")
    start = time.time()
    for f in files:
        with open(f, 'rb') as _: pass
    end = time.time()
    disk_fps = len(files) / (end - start)
    print(f"   -> Disk Speed: {disk_fps:.1f} images/sec")

    # --- TEST 2: CPU DECODE SPEED ---
    print("\n2️⃣  TESTING CPU DECODE SPEED...")
    start = time.time()
    for f in files:
        cv2.imread(f)
    end = time.time()
    decode_fps = len(files) / (end - start)
    print(f"   -> Decode Speed: {decode_fps:.1f} images/sec")

    # --- TEST 3: GPU INFERENCE (STRICTLY LOCAL) ---
    print("\n3️⃣  TESTING GPU AI SPEED...")
    
    # root='.' tells it to look in ./models/
    app = FaceAnalysis(name='buffalo_s', root='.', providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    app.get(dummy) # Warmup

    start = time.time()
    for _ in range(50):
        app.get(dummy)
    end = time.time()
    gpu_fps = 50 / (end - start)
    print(f"   -> AI Inference Speed: {gpu_fps:.1f} FPS")

if __name__ == "__main__":
    benchmark()