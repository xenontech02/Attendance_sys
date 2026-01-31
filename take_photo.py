import cv2
import os

# Create photos directory if it doesn't exist
if not os.path.exists('photos'):
    os.makedirs('photos')

cap = cv2.VideoCapture(0)

print("--- PHOTO CAPTURE MODE ---")
print("1. Look at the camera.")
print("2. Press 's' to SAVE your reference photo.")
print("3. Press 'q' to QUIT.")

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to access camera")
        break
        
    cv2.imshow("Capture Reference Photo", frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Press 's' to save
    if key == ord('s'):
        filename = 'photos/Ali.jpg'
        cv2.imwrite(filename, frame)
        print(f"✅ Success! Saved {filename}")
        print("You can now run main.py")
        break
        
    # Press 'q' to quit
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()