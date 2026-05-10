How to Use It
I have created a unified CLI script (main.py) to easily run the system.

1. Training with Real Data (When you have it)
When you are ready to use the real Roboflow dataset:

bash
python main.py prepare --mode dataset --image-dir "C:/path/to/dataset/images" --label-dir "C:/path/to/dataset/labels"
python main.py train --algorithm random_forest

2. Running Real-Time Detection
To run inference using your webcam (it will load the models I've already trained and saved):

bash
python main.py detect --source 0

3. Running on an Image or Video
To test it on a downloaded video or image:

bash
python main.py detect --source "path/to/video.mp4"
python main.py detect --source "path/to/image.jpg"
Note: The script automatically detects the file type. Annotated outputs will be saved in the output/ directory.