# NCKH_Face_Detection

Installation

1. Clone project
   git clone <your-repo>
   cd project
2. Create virtual environment (recommended)
   python -m venv venv
   venv\Scripts\activate # Windows
3. Install dependencies
   pip install -r requirements.txt
   Pretrained Model

Download RetinaFace pretrained model:

mobilenet0.25_Final.pth

Place it in:

external/retinaface/weights/
Usage

1️. Detect face from image
python src/detect_image.py

Output:

data/output.jpg
2️ .Detect face from webcam (real-time)
python src/detect_webcam.py

Press ESC to exit.

3. Detect face from video
   python src/detect_video.py
4. Register face using webcam
   python src/register_webcam.py

Features:

Auto create folder: user_1, user_2, ...

Capture ~40 images per user

Instructions:

LOOK CENTER

TURN LEFT

TURN RIGHT

LOOK UP

LOOK DOWN

Output:

data/faces/user_x/ 5. Register face from video dataset

Put videos in:

data/video_faces/

Example:

user_01.mp4
user_02.mp4

Run:

python src/register_dataset.py

Output:

data/faces/video_dataset_user_01/
data/faces/video_dataset_user_02/
