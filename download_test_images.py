import os
import urllib.request
import cv2

def download_image(url, save_path):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
            data = response.read()
            out_file.write(data)
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def main():
    img_dir = "data/HardHatDataset/images"
    lbl_dir = "data/HardHatDataset/labels"
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    
    # 1. Image of worker with helmet
    url1 = "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Construction_worker_at_work.jpg/800px-Construction_worker_at_work.jpg"
    
    # 2. Image of person without helmet
    url2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Portrait_of_a_man.jpg/800px-Portrait_of_a_man.jpg"
    
    # 3. Another worker with helmet
    url3 = "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1d/Construction_Worker_on_a_Ladder.jpg/800px-Construction_Worker_on_a_Ladder.jpg"

    # 4. Another person without helmet
    url4 = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a9/Casual_man.jpg/800px-Casual_man.jpg"
    
    images = [
        ("img_0.jpg", url1, 0), # class 0: helmet
        ("img_1.jpg", url2, 1), # class 1: no helmet
        ("img_2.jpg", url3, 0), # class 0: helmet
        ("img_3.jpg", url4, 1), # class 1: no helmet
    ]
    
    for filename, url, label_id in images:
        img_path = os.path.join(img_dir, filename)
        lbl_path = os.path.join(lbl_dir, filename.replace(".jpg", ".txt"))
        
        print(f"Downloading {filename}...")
        if download_image(url, img_path):
            # Create a YOLO label (class_id x y w h)
            # Coordinates don't matter because our script only reads the class_id to determine truth label
            with open(lbl_path, "w") as f:
                f.write(f"{label_id} 0.5 0.5 0.2 0.2\n")
            print(f"Saved {filename} and its label.")

if __name__ == "__main__":
    main()
