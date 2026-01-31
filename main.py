import cv2
import numpy as np
import os
from datetime import datetime

# --- CONFIGURATION ---
path = 'photos'
attendance_file = 'attendance.csv'

# Initialize OpenCV's Native Face Recognizer (No dlib needed!)
recognizer = cv2.face.LBPHFaceRecognizer_create()
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print(f"📂 Scanning folder: {path}")
image_paths = [os.path.join(path, f) for f in os.listdir(path)]
face_samples = []
ids = []
names = {}

# --- TRAIN THE MODEL ---
print("Training system on known faces...")

for image_path in image_paths:
    # Read image in Grayscale (Required for OpenCV)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"⚠️ Could not read {image_path}. Skipping.")
        continue

    # Get the name from the filename (e.g., 'Ali.jpg' -> 'Ali')
    name = os.path.split(image_path)[-1].split(".")[0]
    
    # Create a unique numeric ID for this person (Ali = 1)
    # We use a simple hash of the name for a quick ID
    user_id = abs(hash(name)) % 100000 
    names[user_id] = name

    # Detect face in the photo to learn it
    faces = face_cascade.detectMultiScale(img)

    for (x, y, w, h) in faces:
        face_samples.append(img[y:y+h, x:x+w])
        ids.append(user_id)

if len(face_samples) == 0:
    print("❌ Error: No faces found in 'photos'. Please run take_photo.py again.")
    exit()

recognizer.train(face_samples, np.array(ids))
print(f"✅ Training Complete. Learned faces: {list(names.values())}")

# --- HELPER: MARK ATTENDANCE ---
def markAttendance(name):
    if not os.path.isfile(attendance_file):
        with open(attendance_file, 'w') as f:
            f.write('Name,Time,Date\n')

    with open(attendance_file, 'r+') as f:
        lines = f.readlines()
        nameList = [line.split(',')[0] for line in lines]
        
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            dateString = now.strftime('%Y-%m-%d')
            f.write(f'{name},{dtString},{dateString}\n')
            print(f"📝 Attendance marked for: {name}")

# --- LIVE RECOGNITION LOOP ---
cap = cv2.VideoCapture(0)
print("📷 Camera Started. Press 'q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        # Predict the face
        id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        
        # Confidence logic: 0 is perfect match, >100 is bad match
        if confidence < 80:
            name = names.get(id_, "Unknown").upper()
            markAttendance(name)
            color = (0, 255, 0)
        else:
            name = "UNKNOWN"
            color = (0, 0, 255)

        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, name, (x+5, y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Attendance System (OpenCV)', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()