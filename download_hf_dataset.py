import os
import tqdm
from datasets import load_dataset

def main():
    print("Downloading keremberke/hard-hat-detection from Hugging Face...")
    # Load just the validation split to keep it quick (it has around 300-500 images)
    try:
        dataset = load_dataset("keremberke/hard-hat-detection", "full", split="validation", trust_remote_code=True)
    except Exception as e:
        print(f"Error loading full config: {e}. Trying default...")
        dataset = load_dataset("keremberke/hard-hat-detection", split="validation", trust_remote_code=True)
        
    print(f"Loaded {len(dataset)} images.")
    
    img_dir = "data/HardHatDataset/images"
    lbl_dir = "data/HardHatDataset/labels"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    # Check what features exist
    print("Features:", dataset.features)
    
    # We will process max 500 images
    for idx, item in enumerate(tqdm.tqdm(dataset.select(range(min(500, len(dataset)))))):
        # item['image'] is a PIL image
        img = item["image"]
        
        # Convert to RGB if needed
        if img.mode != "RGB":
            img = img.convert("RGB")
            
        img_path = os.path.join(img_dir, f"img_{idx}.jpg")
        img.save(img_path)
        
        # Write YOLO label file
        lbl_path = os.path.join(lbl_dir, f"img_{idx}.txt")
        with open(lbl_path, "w") as f:
            # The dataset usually has 'objects' dict with 'category', 'bbox'
            if "objects" in item:
                objs = item["objects"]
                categories = objs.get("category", [])
                bboxes = objs.get("bbox", [])
                
                for cat, bbox in zip(categories, bboxes):
                    # cat 0 = hardhat, 1 = no-hardhat
                    # YOLO format: class_id x_center y_center width height (normalized)
                    # We actually only need the class ID in our data_prep.py logic
                    # Our data_prep.py only checks if class 0 or 2 exists in the file.
                    f.write(f"{cat} 0.5 0.5 0.1 0.1\n")
            else:
                # If there are no labels, we just write nothing
                pass

    print(f"Saved {min(500, len(dataset))} images to {img_dir}")

if __name__ == "__main__":
    main()
