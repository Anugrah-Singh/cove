import onnxruntime as ort
import numpy as np
import cv2

def main():
    # Load model
    session = ort.InferenceSession("models/buffalo_s/det_500m.onnx", providers=['CPUExecutionProvider'])
    input_name = session.get_inputs()[0].name
    
    # Create a dummy image (640x640 Black Square)
    # We use a dummy to ensure we see the "Standard" output shapes
    blob = np.random.normal(size=(1, 3, 640, 640)).astype(np.float32)
    
    # Run Model
    print("--- RUNNING DIAGNOSTIC ---")
    outputs = session.run(None, {input_name: blob})
    
    print(f"Model returned {len(outputs)} output layers.")
    
    # Print the shape of every layer
    for i, layer in enumerate(outputs):
        flat_data = layer.flatten()
        print(f"Layer {i}: Shape {layer.shape} | Min: {flat_data.min():.3f} | Max: {flat_data.max():.3f}")
        
        # Heuristic to guess what this layer is
        if layer.shape[1] == 1:
            print(f"   -> Looks like SCORES (Probabilities)")
        elif layer.shape[1] == 4:
            print(f"   -> Looks like BOXES")
        elif layer.shape[1] == 10:
            print(f"   -> Looks like LANDMARKS")

    print("--- END DIAGNOSTIC ---")

if __name__ == "__main__":
    main()
    