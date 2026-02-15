import os
import insightface
from insightface.app import FaceAnalysis
import warnings
import onnxruntime

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class AIEngine:
    def __init__(self, model_name="buffalo_s", root_path="."):
        # Determine available providers
        # available_providers = onnxruntime.get_available_providers()
        # print(f"[INFO] Available ONNX Providers: {available_providers}")
        
        # Force CPU for stability during testing given CUDA 999 errors
        target_providers = ['CPUExecutionProvider']
        print("[INFO] Forcing CPU Execution for Stability")

        # 1. Initialize
        self.app = FaceAnalysis(
            name=model_name, 
            root=root_path, 
            allowed_modules=['detection', 'recognition'], 
            providers=target_providers
        )
        
        # 2. Speed Settings (320x320 is the sweet spot for 200+ FPS)
        # ctx_id=0 for GPU 0, -1 for CPU
        ctx_id = -1
        self.app.prepare(ctx_id=ctx_id, det_size=(320, 320))
        
        # 3. FORCE HEURISTIC SEARCH (The "Instant Start" Fix) - ONE TIME CONFIG
        if ctx_id == 0:
            sess_options = {
                'device_id': 0,
                'gpu_mem_limit': 4 * 1024 * 1024 * 1024,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cudnn_conv_algo_search': 'HEURISTIC', # <--- CRITICAL FIX
                'do_copy_in_default_stream': '1',
            }
            
            try:
                for model in self.app.models.values():
                    if hasattr(model, 'session'):
                        model.session.set_providers(['CUDAExecutionProvider'], [sess_options])
            except Exception as e:
                print(f"[WARNING] Failed to set advanced CUDA options: {e}")
                print("[WARNING] Falling back to default provider settings.")

    def get_faces(self, img):
        return self.app.get(img)