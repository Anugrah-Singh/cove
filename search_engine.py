import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer
import cv2
from PIL import Image

class SearchEngine:
    def __init__(self, model_dir="models"):
        print("Loading CLIP (ONNX)...")
        self.providers = ['CUDAExecutionProvider','CPUExecutionProvider']
        
        # Load the Brains
        try:
            self.img_session = ort.InferenceSession(f"{model_dir}/clip_image.onnx", providers=self.providers)
            self.txt_session = ort.InferenceSession(f"{model_dir}/clip_text.onnx", providers=self.providers)
            self.tokenizer = Tokenizer.from_file(f"{model_dir}/tokenizer.json")
            print("✅ Semantic Search Engine Ready.")
        except Exception as e:
            print("❌ Failed to load CLIP models. Did you run 'download_models.py'?")
            raise e

    def get_image_embedding(self, image_input):
        """
        Generates 512-d vector for an image.
        image_input: Can be a file path (str) OR a PIL Image object.
        """
        try:
            # 1. Load Image
            if isinstance(image_input, str):
                img = Image.open(image_input).convert("RGB")
            else:
                img = image_input.convert("RGB") # It's already a PIL Image

            # 2. Preprocess (Resize/Normalize for CLIP)
            img = img.resize((224, 224))
            img_data = np.array(img).astype(np.float32) / 255.0
            
            # Normalize (CLIP specific mean/std)
            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
            img_data = (img_data - mean) / std
            
            # Transpose to (Batch, Channel, Height, Width) -> (1, 3, 224, 224)
            img_data = np.transpose(img_data, (2, 0, 1))
            img_data = np.expand_dims(img_data, axis=0)

            # 3. Run ONNX Model
            inputs = {self.img_session.get_inputs()[0].name: img_data}
            embedding = self.img_session.run(None, inputs)[0]
            
            # 4. Normalize Vector
            norm = np.linalg.norm(embedding)
            return embedding.flatten() / (norm + 1e-6)

        except Exception as e:
            print(f"Error processing image: {e}")
            return None

    def get_text_embedding(self, text):
        """
        Takes "A dog" -> Returns 512 numbers.
        """
        # 1. Tokenize (with padding/truncation)
        self.tokenizer.enable_padding(length=77) 
        self.tokenizer.enable_truncation(max_length=77)
        
        encoding = self.tokenizer.encode(text)
        
        # Standardize inputs to 64-bit integers
        input_ids = np.array([encoding.ids], dtype=np.int64)
        
        # --- DYNAMIC INPUT MAPPING ---
        # We check what the model actually expects to avoid 'Invalid input name'
        model_inputs = [i.name for i in self.txt_session.get_inputs()]
        
        inputs = {}
        if "input_ids" in model_inputs:
            inputs["input_ids"] = input_ids
            
        # Some models use 'attention_mask', some don't. 
        # We only provide it if the model asks for it.
        if "attention_mask" in model_inputs:
            inputs["attention_mask"] = np.array([encoding.attention_mask], dtype=np.int64)

        # 2. Run ONNX Model
        outputs = self.txt_session.run(None, inputs)
        
        # CLIP Text models usually return two things: 
        # [0] is the hidden states, [1] is the pooled/final embedding (512-d).
        # We want the one that is shape (1, 512).
        if outputs[0].shape[-1] == 512:
            embedding = outputs[0]
        else:
            embedding = outputs[1]
        
        # 3. Normalize Vector
        norm = np.linalg.norm(embedding)
        return embedding.flatten() / (norm + 1e-6)