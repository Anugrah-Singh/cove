import os
import shutil

def finalize_structure():
    print("🔧 Model Structure Finalizer")
    
    # 1. Ensure Destination Exists
    dest_dir = "models/buffalo_l"
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"   -> Created {dest_dir}")

    # --- HANDLE DETECTION MODEL (det_10g.onnx) ---
    target_det = os.path.join(dest_dir, "det_10g.onnx")
    
    # Potential sources for the detection model
    possible_dets = [
        "models/det_10g.onnx",
        "models/face_detection.onnx",
        "models/buffalo_l/det_10g.onnx" # It might already be there
    ]
    
    found_det = False
    for path in possible_dets:
        if os.path.exists(path):
            if path == target_det:
                print("   ✅ Detection model is already in the correct place.")
                found_det = True
                break
            else:
                print(f"   -> Found detection model at: {path}")
                shutil.move(path, target_det)
                print("      Moved to models/buffalo_l/det_10g.onnx")
                found_det = True
                break
    
    if not found_det:
        print("   ❌ ERROR: Could not find 'det_10g.onnx' or 'face_detection.onnx'")

    # --- HANDLE RECOGNITION MODEL (w600k_r50.onnx) ---
    target_rec = os.path.join(dest_dir, "w600k_r50.onnx")
    
    # Potential sources for the recognition model
    possible_recs = [
        "models/w600k_r50.onnx",
        "models/face_recognition.onnx", # You renamed it to this
        "models/buffalo_l/w600k_r50.onnx",
        "models/buffalo_l/face_recognition.onnx"
    ]
    
    found_rec = False
    for path in possible_recs:
        if os.path.exists(path):
            if path == target_rec:
                print("   ✅ Recognition model is already in the correct place.")
                found_rec = True
                break
            else:
                print(f"   -> Found recognition model at: {path}")
                shutil.move(path, target_rec)
                print("      Renamed & Moved to models/buffalo_l/w600k_r50.onnx")
                found_rec = True
                break

    if not found_rec:
        print("   ❌ ERROR: Could not find 'w600k_r50.onnx' or 'face_recognition.onnx'")

    # --- FINAL CHECK ---
    if found_det and found_rec:
        print("\n🎉 SUCCESS! Structure is now valid.")
        print(f"   {target_det}")
        print(f"   {target_rec}")
    else:
        print("\n⚠️ Structure is incomplete. Check the errors above.")

if __name__ == "__main__":
    finalize_structure()