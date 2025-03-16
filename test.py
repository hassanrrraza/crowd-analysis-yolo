import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np

# Ensure that you have GPU support enabled for YOLO
## model = YOLO('best.pt', device='0')  # '0' represents the GPU device index

model = YOLO('best.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)


def draw_counting_bar(frame, current_count, threshold):
    bar_width = 200
    bar_height = 20
    padding = 10
    bar_x = padding
    bar_y = padding
    filled_width = min(int((current_count / threshold) * bar_width), bar_width)  # Ensure not to exceed threshold
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (0, 0, 255), 2)  # Blue border
    cvzone.putTextRect(frame, f'Count: {current_count}', (bar_x + 10, bar_y + 15), 1, 1)


def draw_threshold_bar(frame, current_count, threshold):
    bar_width = 400
    bar_height = 20
    padding = 10
    bar_x = frame.shape[1] - bar_width - padding  # Align to the right side
    bar_y = padding
    percentage = min(int((current_count / threshold) * 100), 100)  # Ensure not to exceed 100%
    filled_width = min(int((current_count / threshold) * bar_width), bar_width)  # Ensure not to exceed threshold
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), (0, 255, 0), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 192, 203), 2)  # Pink border
    cvzone.putTextRect(frame, f'Threshold: {percentage}%', (bar_x - 150, bar_y + 15), 1, 1)


cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('cr.mp4')

my_file = open("coco1.txt", "r")
data = my_file.read()
class_list = data.split("\n")
# print(class_list)

count = 0
area1 = [(463, 147), (426, 460), (89, 428), (14, 222), (249, 105)]

# Set your threshold here
threshold = 40

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    list1 = []

    for index, row in px.iterrows():
        x1 = int(row[0])
        y1 = int(row[1])
        x2 = int(row[2])
        y2 = int(row[3])
        d = int(row[5])
        c = class_list[d]
        cx = int(x1 + x2) // 2
        cy = int(y1 + y2) // 2
        w, h = x2 - x1, y2 - y1
        # area1 code
        result = cv2.pointPolygonTest(np.array(area1, np.int32), ((cx, cy)), False)
        if result >= 0:
            cvzone.cornerRect(frame, (x1, y1, w, h), 3, 2)
            list1.append(cx)

    cr1 = len(list1)

    cv2.polylines(frame, [np.array(area1, np.int32)], True, (0, 0, 255), 2)

    draw_counting_bar(frame, cr1, threshold)
    draw_threshold_bar(frame, cr1, threshold)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()