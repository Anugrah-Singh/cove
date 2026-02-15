import time
import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

def deep_trace():
    print("🔬 DEEP DIAGNOSTIC TRACE")
    print("=" * 50)

    # 1. SETUP: Time how long initialization takes
    print("1️⃣  Initializing Engine...")
    t0 = time.time()
    app = FaceAnalysis(name='buffalo_s', root='.', allowed_modules=['detection', 'recognition'], providers=['CUDAExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(320, 320))
    print(f"   -> Init Time: {time.time() - t0:.4f}s")

    # 2. DATA: Pick 3 real images
    img_dir = "test_images"
    files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith('.jpg')][:3]
    
    if not files:
        print("❌ No images found to test!")
        return

    print("\n2️⃣  TRACING REAL IMAGES (Where is the lag?)")
    print(f"   {'Filename':<20} | {'Res':<12} | {'Load(ms)':<10} | {'AI(ms)':<10} | {'Total(ms)':<10} | {'FPS':<6}")
    print("-" * 85)

    for f in files:
        # TIMING BLOCK
        t_start = time.time()
        
        # A. Disk Read
        img = cv2.imread(f)
        t_load = time.time()
        
        # B. AI Inference
        faces = app.get(img)
        t_ai = time.time()
        
        # Calc Deltas
        ms_load = (t_load - t_start) * 1000
        ms_ai = (t_ai - t_load) * 1000
        ms_total = (t_ai - t_start) * 1000
        fps = 1.0 / (t_ai - t_start)
        
        h, w = img.shape[:2]
        filename = os.path.basename(f)
        if len(filename) > 18: filename = filename[:15] + "..."
        
        print(f"   {filename:<20} | {w}x{h:<7} | {ms_load:>8.2f}   | {ms_ai:>8.2f}   | {ms_total:>8.2f}   | {fps:>5.1f}")

    print("\n3️⃣  CONTROL TEST (Synthetic Data)")
    # Compare against a perfect 640x640 black square
    dummy = np.zeros((640, 640, 3), dtype=np.uint8)
    
    t_start = time.time()
    app.get(dummy)
    t_end = time.time()
    
    ms_dummy = (t_end - t_start) * 1000
    print(f"   {'Dummy_640x640':<20} | {'640x640':<12} | {'0.00':<10} | {ms_dummy:>8.2f}   | {ms_dummy:>8.2f}   | {1.0/(t_end-t_start):>5.1f}")

if __name__ == "__main__":
    deep_trace()