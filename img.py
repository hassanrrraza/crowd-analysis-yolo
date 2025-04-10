import cv2
import time
import os

# Create dataset directory if it doesn't exist
dataset_dir = "dataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

cpt = 0
maxFrames = 100  # Extract 100 frames

cap = cv2.VideoCapture('abb.mp4')
while cpt < maxFrames:
    ret, frame = cap.read()
    if not ret:
        break
#    count += 1
#    if count % 3 != 0:
#        continue
    frame = cv2.resize(frame, (1080, 500))
    cv2.imshow("test window", frame)  # show image in window
    cv2.imwrite(os.path.join(dataset_dir, "person_%d.jpg" % cpt), frame)
    time.sleep(0.01)
    cpt += 1
    if cv2.waitKey(5) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()