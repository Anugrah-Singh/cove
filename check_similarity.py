import cv2
import numpy as np
from ai_engine import AIEngine

def main():
    # 1. Initialize
    ai = AIEngine()
    
    # 2. Pick two photos that are DIFFERENT people
    img_path1 = "test_images/Aaron_Peirsol_0001.jpg"
    img_path2 = "test_images/Abdoulaye_Wade_0001.jpg"
    
    # 3. Load & Process
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    
    if img1 is None or img2 is None:
        print("Error: Check image paths.")
        return

    # 4. Get Vectors
    print("Extracting Face 1...")
    faces1 = ai.get_faces(img1)
    if not faces1:
        print("No face found in Image 1")
        return
    vec1 = faces1[0].embedding
    
    print("Extracting Face 2...")
    faces2 = ai.get_faces(img2)
    if not faces2:
        print("No face found in Image 2")
        return
    vec2 = faces2[0].embedding
    
    # 5. Calculate Similarity (Cosine)
    vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-6)
    vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-6)
    sim = float(np.dot(vec1_norm, vec2_norm))
    distance = 1.0 - sim
    
    print(f"\n--- RESULTS ---")
    print(f"Similarity Score: {sim:.4f} (Higher = More Similar)")
    print(f"Cosine Distance:  {distance:.4f} (Lower = More Similar)")
    print("------------------------------------------------")
    
    if distance < 0.25:
        print("🚨 The AI thinks these are the SAME PERSON!")
        print("   -> You need to LOWER your threshold (eps) below this distance.")
    else:
        print("✅ The AI knows these are DIFFERENT people.")
        print("   -> Your threshold (eps) should be near this value.")

if __name__ == "__main__":
    main()