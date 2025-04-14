import cv2
import os

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        print(f"Coordinate: ({x}, {y})")
        # Draw a point on the frame
        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)
        cv2.imshow("Get Coordinates", frame)

# Get paths relative to script location
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
video_path = os.path.join(base_dir, "data", "videos", "cr.mp4")

# Fallback to root directory if not found
if not os.path.exists(video_path):
    video_path = os.path.join(base_dir, "cr.mp4")

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Create window and set mouse callback
cv2.namedWindow("Get Coordinates")
cv2.setMouseCallback("Get Coordinates", mouse_callback)

print("Instructions:")
print("1. Click on the video frame to get coordinates")
print("2. Press 'n' to move to next frame")
print("3. Press 'ESC' to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning of video
        continue
        
    frame = cv2.resize(frame, (1280, 720))  # Resize to match your main script
    
    # Display the frame
    cv2.imshow("Get Coordinates", frame)
    
    # Wait for key press
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC key
        break
    elif key == ord('n'):  # Next frame
        continue

cap.release()
cv2.destroyAllWindows() 