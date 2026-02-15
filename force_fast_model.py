import shutil
import os
import insightface
from insightface.app import FaceAnalysis

def force_switch():
    print("🚀 VISION ARCHIVE: Force Switch to 'Small' Model")
    
    # 1. DELETE THE OLD HEAVY MODELS
    slow_path = "models/buffalo_l"
    if os.path.exists(slow_path):
        print(f"   🗑️ Deleting {slow_path} (ResNet50)...")
        shutil.rmtree(slow_path)
    
    # 2. DELETE CACHE CONFUSION
    # Sometimes InsightFace downloads to ~/.insightface/models/
    # We want it LOCAL.
    
    # 3. DOWNLOAD 'buffalo_s' (MobileFaceNet)
    print("   ⬇️ Downloading 'buffalo_s' (MobileFaceNet)...")
    
    # We set root='.' so it installs into ./models/buffalo_s
    app = FaceAnalysis(name='buffalo_s', root='.', providers=['CUDAExecutionProvider'])
    
    # Verify installation
    expected_path = "models/buffalo_s"
    if os.path.exists(expected_path):
        print(f"   ✅ Success! Installed to {expected_path}")
        print("      - w600k_mbf.onnx (Recognition - Fast)")
        print("      - det_500m.onnx (Detection - Fast)")
    else:
        print("   ❌ Error: Model downloaded but path is unexpected. Check './models'")

if __name__ == "__main__":
    force_switch()
    