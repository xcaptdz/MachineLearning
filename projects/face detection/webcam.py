import cv2
import os

# Create a directory to save snapshots if it doesn't exist
output_dir = "snapshots"
os.makedirs(output_dir, exist_ok=True)

# Load the Haar Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize the webcam (0 for default camera)
cap = cv2.VideoCapture(0)

snapshot_count = 0  # Counter for snapshots

print("Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale (face detection works better on grayscale images)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    # Draw rectangles around detected faces and save snapshots
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Save snapshot with a unique name
        snapshot_filename = os.path.join(output_dir, f'snapshot_{snapshot_count}.png')
        cv2.imwrite(snapshot_filename, frame)
        print(f"Snapshot saved: {snapshot_filename}")
        
        snapshot_count += 1

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
