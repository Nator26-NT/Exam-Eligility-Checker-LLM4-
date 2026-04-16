# config.py (lightweight version)
import os

BASE_DIR = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}

# Only face cascade is needed
FACE_CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")

# Detection settings
MIN_FACE_SIZE = (60, 60)
CAMERA_INDEX = 0