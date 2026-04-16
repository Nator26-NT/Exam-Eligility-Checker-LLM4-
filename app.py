from flask import Flask, render_template, request, redirect, url_for, Response, jsonify
import cv2
import os
import base64
import numpy as np
import config
from models import CVModels
from checker import EligibilityChecker

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# Initialize CV components once
models = CVModels(config)
checker = EligibilityChecker(config, models)

# ------------------- Helper: process image -------------------
def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        return False, "Could not read image", None
    eligible, reason, annotated = checker.check(frame)
    out_path = os.path.join(config.UPLOAD_FOLDER, "result.jpg")
    cv2.imwrite(out_path, annotated)
    return eligible, reason, out_path

# ------------------- Helper: process video -------------------
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    results = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % 15 == 0:
            eligible, reason, _ = checker.check(frame)
            results.append((eligible, reason))
        frame_count += 1
    cap.release()
    if not results:
        return False, "No frames processed"
    for eligible, reason in results:
        if not eligible:
            return False, reason
    return True, "All frames eligible"

# ------------------- Routes -------------------
@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Exam Eligibility Checker</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            body {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                min-height: 100vh;
                padding: 2rem;
                transition: background 0.3s, color 0.3s;
            }
            body.light {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                color: #1a1a2e;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            h1 {
                text-align: center;
                margin-bottom: 0.5rem;
                font-size: 2.5rem;
                background: linear-gradient(135deg, #00b4db, #0083b0);
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
            }
            .sub {
                text-align: center;
                color: #8899aa;
                margin-bottom: 2rem;
            }
            .cards {
                display: flex;
                flex-wrap: wrap;
                gap: 2rem;
                justify-content: center;
            }
            .card {
                background: rgba(255,255,255,0.1);
                backdrop-filter: blur(10px);
                border-radius: 20px;
                padding: 1.5rem;
                width: 300px;
                text-align: center;
                transition: transform 0.3s, box-shadow 0.3s;
                box-shadow: 0 8px 20px rgba(0,0,0,0.2);
            }
            body.light .card {
                background: rgba(255,255,255,0.9);
                box-shadow: 0 8px 20px rgba(0,0,0,0.1);
            }
            .card:hover {
                transform: translateY(-10px);
                box-shadow: 0 15px 30px rgba(0,0,0,0.3);
            }
            .card h3 {
                margin: 1rem 0;
                font-size: 1.5rem;
            }
            .card p {
                color: #aaa;
                margin-bottom: 1.5rem;
            }
            body.light .card p {
                color: #555;
            }
            .btn {
                background: #00b4db;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 30px;
                cursor: pointer;
                font-size: 1rem;
                transition: background 0.3s;
                text-decoration: none;
                display: inline-block;
            }
            .btn:hover {
                background: #0083b0;
            }
            input[type="file"] {
                margin: 1rem 0;
                display: block;
                width: 100%;
            }
            .result-area {
                margin-top: 2rem;
                text-align: center;
                background: rgba(0,0,0,0.5);
                border-radius: 15px;
                padding: 1rem;
            }
            body.light .result-area {
                background: rgba(0,0,0,0.1);
            }
            .spinner {
                display: none;
                margin: 1rem auto;
                width: 40px;
                height: 40px;
                border: 4px solid #f3f3f3;
                border-top: 4px solid #00b4db;
                border-radius: 50%;
                animation: spin 1s linear infinite;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .theme-toggle {
                position: fixed;
                top: 20px;
                right: 20px;
                background: rgba(255,255,255,0.2);
                border: none;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                cursor: pointer;
                font-size: 1.5rem;
                backdrop-filter: blur(5px);
            }
            footer {
                text-align: center;
                margin-top: 3rem;
                color: #8899aa;
            }
            video, img {
                max-width: 100%;
                border-radius: 10px;
                margin-top: 1rem;
            }
        </style>
    </head>
    <body>
        <button class="theme-toggle" onclick="toggleTheme()">🌓</button>
        <div class="container">
            <h1>📝 Exam Eligibility Checker</h1>
            <div class="sub">AI-powered proctoring assistant</div>
            <div class="cards">
                <div class="card">
                    <h3>📸 Upload Photo</h3>
                    <p>Submit a single image of your exam room</p>
                    <form id="photoForm" enctype="multipart/form-data">
                        <input type="file" name="file" accept="image/*" required>
                        <button type="submit" class="btn">Check Photo</button>
                    </form>
                    <div id="photoSpinner" class="spinner"></div>
                    <div id="photoResult"></div>
                </div>
                <div class="card">
                    <h3>🎥 Upload Video</h3>
                    <p>Upload a video recording of your room</p>
                    <form id="videoForm" enctype="multipart/form-data">
                        <input type="file" name="file" accept="video/*" required>
                        <button type="submit" class="btn">Check Video</button>
                    </form>
                    <div id="videoSpinner" class="spinner"></div>
                    <div id="videoResult"></div>
                </div>
                <div class="card">
                    <h3>🎥 Live Webcam</h3>
                    <p>Real-time face & object detection</p>
                    <button class="btn" onclick="location.href='/webcam'">Start Webcam Check</button>
                </div>
            </div>
            <footer>
                Ensure only one face is visible, no phones/books/laptops/papers.
            </footer>
        </div>
        <script>
            function toggleTheme() {
                document.body.classList.toggle('light');
            }
            // Handle photo upload via fetch
            const photoForm = document.getElementById('photoForm');
            if(photoForm) {
                photoForm.addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const formData = new FormData(photoForm);
                    const spinner = document.getElementById('photoSpinner');
                    const resultDiv = document.getElementById('photoResult');
                    spinner.style.display = 'block';
                    resultDiv.innerHTML = '';
                    try {
                        const resp = await fetch('/upload_photo_ajax', { method: 'POST', body: formData });
                        const data = await resp.json();
                        spinner.style.display = 'none';
                        resultDiv.innerHTML = `<div class="result-area"><h3>${data.eligible ? '✅ ELIGIBLE' : '❌ NOT ELIGIBLE'}</h3><p>${data.reason}</p><img src="data:image/jpeg;base64,${data.image}" width="100%"></div>`;
                    } catch(err) {
                        spinner.style.display = 'none';
                        resultDiv.innerHTML = '<div class="result-area">Error processing image</div>';
                    }
                });
            }
            // Video upload via fetch
            const videoForm = document.getElementById('videoForm');
            if(videoForm) {
                videoForm.addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const formData = new FormData(videoForm);
                    const spinner = document.getElementById('videoSpinner');
                    const resultDiv = document.getElementById('videoResult');
                    spinner.style.display = 'block';
                    resultDiv.innerHTML = '';
                    try {
                        const resp = await fetch('/upload_video_ajax', { method: 'POST', body: formData });
                        const data = await resp.json();
                        spinner.style.display = 'none';
                        resultDiv.innerHTML = `<div class="result-area"><h3>${data.eligible ? '✅ ELIGIBLE' : '❌ NOT ELIGIBLE'}</h3><p>${data.reason}</p></div>`;
                    } catch(err) {
                        spinner.style.display = 'none';
                        resultDiv.innerHTML = '<div class="result-area">Error processing video</div>';
                    }
                });
            }
        </script>
    </body>
    </html>
    '''

@app.route('/upload_photo_ajax', methods=['POST'])
def upload_photo_ajax():
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'eligible': False, 'reason': 'No file provided'})
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    eligible, reason, annotated_path = process_image(path)
    with open(annotated_path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    return jsonify({'eligible': eligible, 'reason': reason, 'image': img_b64})

@app.route('/upload_video_ajax', methods=['POST'])
def upload_video_ajax():
    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({'eligible': False, 'reason': 'No file provided'})
    path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(path)
    eligible, reason = process_video(path)
    return jsonify({'eligible': eligible, 'reason': reason})

@app.route('/webcam')
def webcam():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live Webcam Eligibility</title>
        <style>
            body {
                background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                color: white;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                text-align: center;
                padding: 2rem;
            }
            video, canvas, img {
                border-radius: 15px;
                margin: 1rem;
                box-shadow: 0 10px 25px rgba(0,0,0,0.3);
            }
            button {
                background: #00b4db;
                border: none;
                padding: 12px 24px;
                border-radius: 40px;
                font-size: 1.2rem;
                cursor: pointer;
                margin: 0.5rem;
                transition: transform 0.2s;
            }
            button:hover {
                transform: scale(1.05);
            }
            .result {
                font-size: 1.5rem;
                margin: 1rem;
                padding: 1rem;
                border-radius: 15px;
                background: rgba(0,0,0,0.5);
            }
            .back {
                margin-top: 2rem;
                display: inline-block;
                background: #555;
                padding: 10px 20px;
                border-radius: 30px;
                text-decoration: none;
                color: white;
            }
        </style>
    </head>
    <body>
        <h1>📹 Live Webcam Check</h1>
        <video id="video" width="640" height="480" autoplay playsinline></video>
        <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
        <br>
        <button id="captureBtn">📸 Capture & Check</button>
        <div id="result" class="result"></div>
        <img id="resultImg" width="640">
        <br>
        <a href="/" class="back">← Back to Home</a>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const resultDiv = document.getElementById('result');
            const resultImg = document.getElementById('resultImg');
            const captureBtn = document.getElementById('captureBtn');

            navigator.mediaDevices.getUserMedia({ video: true })
                .then(stream => { video.srcObject = stream; })
                .catch(err => { resultDiv.innerText = 'Camera access denied'; });

            captureBtn.onclick = async () => {
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                canvas.toBlob(async (blob) => {
                    let formData = new FormData();
                    formData.append('frame', blob, 'frame.jpg');
                    resultDiv.innerHTML = '<div class="spinner"></div>';
                    try {
                        let resp = await fetch('/check_frame', { method: 'POST', body: formData });
                        let data = await resp.json();
                        resultDiv.innerHTML = `<strong>${data.eligible ? '✅ ELIGIBLE' : '❌ NOT ELIGIBLE'}</strong><br>${data.reason}`;
                        if (data.image) resultImg.src = 'data:image/jpeg;base64,' + data.image;
                    } catch(e) {
                        resultDiv.innerHTML = 'Error processing frame';
                    }
                }, 'image/jpeg');
            };
        </script>
    </body>
    </html>
    '''

@app.route('/check_frame', methods=['POST'])
def check_frame():
    file = request.files['frame']
    if not file:
        return jsonify({'eligible': False, 'reason': 'No frame received'})
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    eligible, reason, annotated = checker.check(frame)
    _, buffer = cv2.imencode('.jpg', annotated)
    img_b64 = base64.b64encode(buffer).decode()
    return jsonify({'eligible': eligible, 'reason': reason, 'image': img_b64})

if __name__ == '__main__':
    app.run(debug=True)