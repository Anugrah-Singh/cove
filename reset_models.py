import os
import shutil
import urllib.request
import zipfile

def reset_models():
    print("♻️ VISION ARCHIVE: Factory Reset Models")
    
    target_dir = "models/buffalo_l"
    
    # 1. DELETE EXISTING (Corrupted?) FILES
    if os.path.exists(target_dir):
        print(f"   -> Deleting old {target_dir}...")
        shutil.rmtree(target_dir)
    
    os.makedirs(target_dir)

    # 2. DOWNLOAD OFFICIAL RELEASE (Buffalo_L)
    # This zip contains the exact det_10g.onnx and w600k_r50.onnx our code needs.
    url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    zip_path = "models/buffalo_l.zip"
    
    print("   -> Downloading official InsightFace models (280MB)...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"   ❌ Download failed: {e}")
        return

    print("   -> Unzipping...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
    
    # Cleanup
    os.remove(zip_path)
    
    # 3. VERIFY
    files = os.listdir(target_dir)
    if "det_10g.onnx" in files and "w600k_r50.onnx" in files:
        print("✅ SUCCESS! Official models installed.")
        print(f"   Folder content: {files}")
    else:
        print("⚠️ Warning: Download finished but files look wrong.")
        print(f"   Folder content: {files}")

if __name__ == "__main__":
    reset_models()