import os
import shutil

def main():
    # 1. Source Directory (Where your LFW folders are)
    # Based on your screenshot, it seems to be here:
    source_root = "image_dataset/archive/lfw-deepfunneled/lfw-deepfunneled"
    
    # 2. Destination Directory (Where we want them)
    dest_dir = "test_images"
    
    # Create destination if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created folder: {dest_dir}")
    
    # 3. Walk through the source folders
    print(f"Scanning {source_root}...")
    
    count = 0
    # os.walk goes into every subfolder automatically
    for root, dirs, files in os.walk(source_root):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # Get the full path of the current image
                src_path = os.path.join(root, file)
                
                # Create the destination path
                dst_path = os.path.join(dest_dir, file)
                
                # Check if file already exists to prevent overwriting
                if os.path.exists(dst_path):
                    # Rename if duplicate exists (e.g., Aaron_001_copy.jpg)
                    base, ext = os.path.splitext(file)
                    dst_path = os.path.join(dest_dir, f"{base}_dup{ext}")
                
                # Copy the file
                shutil.copy2(src_path, dst_path)
                
                count += 1
                if count % 100 == 0:
                    print(f"Copied {count} images...", end='\r')

    print(f"\n✅ Done! Copied {count} images to '{dest_dir}'.")
    print("You can now run 'python test_pipeline.py' to test the full cluster engine.")

if __name__ == "__main__":
    # Check if the source path actually exists before running
    if not os.path.exists("image_dataset/archive/lfw-deepfunneled/lfw-deepfunneled"):
        print("❌ Error: Could not find the LFW folder.")
        print("Please check the path inside the script and ensure it matches your folder structure.")
    else:
        main()