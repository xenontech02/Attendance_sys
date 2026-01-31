# Attendance_sys
A real-time facial recognition system that automates attendance logging using Computer Vision. Built with Python and OpenCV's native LBPH recognizer for high efficiency and offline capability.

## 🚀 Features
- **Real-Time Detection:** Instantly detects faces via webcam feed.
- **Automated Logging:** Saves attendance to a `attendance.csv` file with precise timestamps.
- **Duplicate Prevention:** Ensures a student is only marked once per session to avoid spamming the logs.
- **Visual Feedback:** Displays a green bounding box and the student's name when a match is found.

## 🛠️ Tech Stack
- **Language:** Python 3.12
- **Computer Vision:** OpenCV (`cv2`)
- **Algorithm:** LBPH (Local Binary Patterns Histograms) Face Recognizer
- **Data Handling:** NumPy, CSV
