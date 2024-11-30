import cv2
import os



entrance = "rtsp://admin:Welcome2009$@10.0.20.70:554/cam/realmonitor?channel=1&subtype=0"
doorbell = "rtsp://xcapt:Welcome2009$@10.0.20.74:554/h264Preview_01_main"
hall = "rtsp://admin:IYWIVM@10.0.20.72:554/H.264"

# RTSP stream URL (replace with your camera's RTSP URL)
rtsp_url = entrance




# Create a directory to save snapshots
output_dir = "rtsp_snapshots"
os.makedirs(output_dir, exist_ok=True)

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

snapshot_count = 0  # Counter for snapshots

if not cap.isOpened():
    print("Error: Unable to open RTSP stream.")
    exit()

print("Press 'q' to quit.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Convert frame to grayscale for better face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces and save snapshots
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Save the frame with detected faces
        snapshot_filename = os.path.join(output_dir, f'snapshot_{snapshot_count}.png')
        cv2.imwrite(snapshot_filename, frame)
        print(f"Snapshot saved: {snapshot_filename}")

        snapshot_count += 1

    # Display the frame with detected faces
    cv2.imshow('RTSP Face Detection', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
