import cv2
import numpy as np
import math

def find_available_camera(max_index=10):
    """Finds the first available camera index."""
    for index in range(max_index):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            print(f"✅ Camera found at index: {index}")
            return index
        cap.release()
    print("🚫 No available camera found.")
    return -1

# Detect working camera
camera_index = find_available_camera()
if camera_index == -1:
    print("❌ Unable to access any webcam. Please check camera connection or drivers.")
    exit()

# Start capturing
cap = cv2.VideoCapture(camera_index)

while True:
    ret, img = cap.read()
    if not ret or img is None:
        print("⚠️ Frame not captured. Retrying...")
        continue

    cv2.rectangle(img, (50, 50), (400, 400), (0, 255, 0), 2)
    crop_img = img[50:400, 50:400]

    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (35, 35), 0)
    _, thresh1 = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cv2.imshow('Thresholded', thresh1)

    contours, _ = cv2.findContours(thresh1.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        continue

    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    hull = cv2.convexHull(cnt, returnPoints=False)
    if hull is None or len(hull) < 3:
        continue

    defects = cv2.convexityDefects(cnt, hull)
    if defects is None:
        continue

    count_defects = 0
    for i in range(defects.shape[0]):
        s, e, f, _ = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])

        a = np.linalg.norm(np.array(start) - np.array(end))
        b = np.linalg.norm(np.array(start) - np.array(far))
        c = np.linalg.norm(np.array(end) - np.array(far))

        if b * c == 0:
            continue

        angle = math.acos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_img, far, 5, [0, 0, 255], -1)
        cv2.line(crop_img, start, end, [0, 255, 0], 2)

    gestures = {
        0: "Hello World!!!",
        1: "GESTURE ONE",
        2: "GESTURE TWO",
        3: "GESTURE THREE",
        4: "GESTURE FOUR",
        5: "GESTURE FIVE"
    }

    gesture_text = gestures.get(count_defects, "Unknown")
    cv2.putText(img, gesture_text, (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
    cv2.putText(crop_img, f"{count_defects} Fingers", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    drawing = np.zeros(crop_img.shape, np.uint8)
    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 2)
    cv2.drawContours(drawing, [cv2.convexHull(cnt)], 0, (0, 0, 255), 2)

    cv2.imshow('Gesture', img)
    cv2.imshow('Contours', np.hstack((drawing, crop_img)))

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

cap.release()
cv2.destroyAllWindows()
