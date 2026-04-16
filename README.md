# Exam Eligibility Checker

A computer vision web application that verifies if a candidate is eligible to take an exam based on real‑time or uploaded media.  
It checks for **exactly one face** and **prohibited objects** (phone, book, laptop, paper) anywhere in the room – even far from the learner.

## Features

- **Three input modes**  
  - Upload a photo of the exam room  
  - Upload a video recording  
  - Live webcam check  
- **Automatic eligibility decision** with clear reason  
- **Annotated output** – bounding boxes around faces (green) and prohibited objects (red)  
- **Modern GUI** with dark/light theme, spinners, and AJAX uploads  
- **Runs entirely offline** – no cloud API calls  
- **Lightweight** – uses only OpenCV + Flask (no heavy deep learning frameworks)

## Project Structure (4 Python files)
