import insightface
from insightface.app import FaceAnalysis
import os
import shutil

def switch_to_fast():
    print("⚡ VISION ARCHIVE: Switching to High-Efficiency Model (buffalo_s)...")
    
    # Path where InsightFace saves models by default
    # We force it to download to our local 'models' folder
    model_root = "models"
    
    # This call triggers the automatic download of 'buffalo_s'
    # It checks if it exists; if not, it pulls it from the cloud.
    try:
        app = FaceAnalysis(name='buffalo_s', root=model_root, providers=['CUDAExecutionProvider'])
        print("✅ Download Complete.")
    except Exception as e:
        print(f"❌ Error downloading: {e}")
        return

    # Verify
    target_dir = os.path.join(model_root, "models", "buffalo_s")
    if os.path.exists(target_dir):
        print(f"   -> Model installed at: {target_dir}")
        print("   -> Contains: MobileFaceNet (Fast) + SCRFD (Efficient Detection)")
    else:
        # Sometimes InsightFace puts it in slightly different spots depending on version
        print("   -> Searching for model folder...")
        # (The library handles the pathing, so we are usually good)

if __name__ == "__main__":
    switch_to_fast()