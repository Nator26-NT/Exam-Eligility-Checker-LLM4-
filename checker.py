import cv2

class EligibilityChecker:
    def __init__(self, config, models):
        self.config = config
        self.models = models
    
    def check(self, frame):
        faces = self.models.detect_faces(frame)
        all_objects = self.models.detect_objects(frame)
        
        # Any detected object is considered prohibited
        prohibited = all_objects   # all are "cell phone", "book", or "paper"
        
        # Annotate frame
        annotated = frame.copy()
        for (x,y,w,h) in faces:
            cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,255,0), 2)
        for (name, _, conf, (x,y,w,h)) in prohibited:
            cv2.rectangle(annotated, (x,y), (x+w,y+h), (0,0,255), 2)
            cv2.putText(annotated, f"{name} {conf:.2f}", (x,y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        
        # Rules
        face_count = len(faces)
        if face_count == 0:
            return False, "No face detected", annotated
        if face_count > 1:
            return False, f"Multiple faces ({face_count}) detected", annotated
        if prohibited:
            items = ', '.join(set([p[0] for p in prohibited]))
            return False, f"Prohibited object(s): {items}", annotated
        
        # Face size check (closeness)
        (x,y,w,h) = faces[0]
        if w < 80 or h < 80:
            return False, "Face too small – move closer", annotated
        
        return True, "Eligible – ready for exam", annotated