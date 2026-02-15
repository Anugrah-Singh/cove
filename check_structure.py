import os

def check():
    print("🔍 VISION ARCHIVE: Model Auditor")
    print("-" * 40)
    
    root = "models"
    if not os.path.exists(root):
        print(f"❌ '{root}' folder does not exist!")
        return

    found_fast = False
    found_slow = False

    for dirpath, dirnames, filenames in os.walk(root):
        level = dirpath.replace(root, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print(f"{indent}📂 {os.path.basename(dirpath)}/")
        subindent = ' ' * 4 * (level + 1)
        
        for f in filenames:
            if f.endswith(".onnx"):
                size_mb = os.path.getsize(os.path.join(dirpath, f)) / (1024 * 1024)
                print(f"{subindent}📄 {f} ({size_mb:.1f} MB)")
                
                # Check based on file size/name
                if "w600k" in f: # ResNet50 (Large)
                    found_slow = True
                    print(f"{subindent}   ⚠️ SLOW MODEL DETECTED!")
                if "2d106det" in f or "1k3d68" in f:
                    print(f"{subindent}   ⚠️ UNUSED MODEL (Wasting VRAM)")
    
    print("-" * 40)
    if found_slow:
        print("🚨 DIAGNOSIS: The SLOW model (buffalo_l) is still present.")
    else:
        print("✅ DIAGNOSIS: Only fast models found.")

if __name__ == "__main__":
    check()