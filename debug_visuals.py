import cv2
import numpy as np
from ai_engine import AIEngine

def main():
    ai = AIEngine()
    
    img_path = "test_images/Aaron_Peirsol_0001.jpg" 
    img = cv2.imread(img_path)
    
    if img is None:
        print("Error: Image not found.")
        return

    print("Running detection...")
    faces = ai.get_faces(img)
    
    if not faces:
        print("No faces found!")
        return

    print(f"Found {len(faces)} faces. Drawing the first one...")
    face = faces[0]
    
    # 1. Draw Bounding Box (Green)
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # 2. Draw Landmarks (Red Circles)
    # These are the 5 points: Left Eye, Right Eye, Nose, Left Mouth, Right Mouth
    if hasattr(face, 'kps') and face.kps is not None:
        for i, point in enumerate(face.kps):
            px, py = int(point[0]), int(point[1])
            cv2.circle(img, (px, py), 3, (0, 0, 255), -1)
            cv2.putText(img, str(i), (px+5, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # 3. Save result
    cv2.imwrite("debug_visual_check.jpg", img)
    print("✅ Saved 'debug_visual_check.jpg'. Open it and look at the dots!")

if __name__ == "__main__":
    main()