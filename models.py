import cv2
import numpy as np

class CVModels:
    def __init__(self, config):
        self.config = config
        self.face_cascade = cv2.CascadeClassifier(config.FACE_CASCADE_PATH)
        # Background subtractor for moving object detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=36)
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=self.config.MIN_FACE_SIZE
        )
        return faces
    
    def detect_objects(self, frame):
        """
        Detect potential prohibited objects using motion/contour analysis.
        Returns list of (name, class_id, confidence, (x,y,w,h))
        """
        fgmask = self.bg_subtractor.apply(frame)
        # Remove shadows
        fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)[1]
        # Find contours
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        h, w = frame.shape[:2]
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 2000 or area > 50000:   # filter too small/large
                continue
            x, y, cw, ch = cv2.boundingRect(cnt)
            # Aspect ratio filter (phone ~ 0.3-0.6, book ~ 0.7-1.2)
            aspect = cw / float(ch)
            # Simple classification
            if aspect < 0.7:
                name = "cell phone"
            elif aspect < 1.5:
                name = "book"
            else:
                name = "paper"
            confidence = min(1.0, area / 30000.0)   # rough confidence
            objects.append((name, -1, confidence, (x, y, cw, ch)))
        return objects