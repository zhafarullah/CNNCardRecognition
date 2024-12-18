import tensorflow as tf
import cv2
import numpy as np
import os
import time
from collections import deque, Counter

model = tf.keras.models.load_model("64x3-cards.h5")

classes = [i for i in os.listdir(r"D:\TUGAS KULIAHHH\SEM 5\12 2024\CNN train and test\train") if os.path.isdir(os.path.join(r"D:\TUGAS KULIAHHH\SEM 5\12 2024\CNN train and test\train", i))]

frame_buffer = deque(maxlen=90)

def prepare_card(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return img.reshape(-1, 128, 128, 1)

def DrawCircle(image, y, x):
    center_coor = (x, y)
    radius = 10
    color = (0, 0, 255)
    thickness = 2
    image = cv2.circle(image, center_coor, radius, color, thickness)
    return image

def force_vertical(box):
    width = int(np.linalg.norm(box[0] - box[1]))
    height = int(np.linalg.norm(box[1] - box[2]))
    if width > height:
        box = np.roll(box, 1, axis=0)
    return box

def save_top_cards():
    card_counts = Counter(frame_buffer)
    filtered_cards = [card for card, count in card_counts.items() if count >= 5]
    top_cards = sorted(filtered_cards, key=lambda x: -card_counts[x])[:10]
    
    with open("detected_card.txt", "w") as f:
        for card in top_cards:
            f.write(f"{card}\n")
    print(f"File detected_card.txt dibuat dengan kartu:\n" + "\n".join(top_cards))

main_frame_position = (0, 150)  
card_window_offset_x = 0   
card_window_position_y = 500  

fixed_width = 230
fixed_height = 352

cv2.namedWindow('frame', cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('frame', 640, 480)
cv2.moveWindow('frame', *main_frame_position)

cam = cv2.VideoCapture(1)
if not cam.isOpened():
    print("Error opening camera")
    exit()

active_windows = {}
inactive_window_times = {}

while True:
    ret, frame = cam.read()
    if not ret:
        print("Error in retrieving frame")
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([35, 20, 40])
    upper = np.array([80, 255, 255])

    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.bitwise_not(mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    num_labels, labels_im = cv2.connectedComponents(mask)
    min_area = 5000

    current_active_windows = {}

    for i in range(1, num_labels):
        b, k = np.where(labels_im == i)
        pts = np.column_stack((k, b))

        if len(pts) < min_area:
            continue

        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        box = force_vertical(box)

        dst_points = np.array([[0, 0], [fixed_width, 0], [fixed_width, fixed_height], [0, fixed_height]], dtype="float32")
        h_matrix = cv2.getPerspectiveTransform(np.float32(box), dst_points)
        straightened_card = cv2.warpPerspective(frame, h_matrix, (fixed_width, fixed_height))

        card_for_model = prepare_card(straightened_card)
        prediction = model.predict(card_for_model)

        predicted_class_index = int(np.argmax(prediction))
        predicted_class = classes[predicted_class_index]

        frame_buffer.append(predicted_class)

        confidence = prediction[0][predicted_class_index] * 100  
        label = f"{predicted_class} ({confidence:.2f}%)"

        window_name = f"Kartu {predicted_class}"
        cv2.namedWindow(window_name,  cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(window_name, fixed_width, fixed_height)
        
        card_window_start_x = main_frame_position[0] + card_window_offset_x
        card_window_start_y = main_frame_position[1] + card_window_position_y
        cv2.moveWindow(window_name, card_window_start_x, card_window_start_y)
        current_active_windows[window_name] = time.time()

        if window_name not in active_windows:
            cv2.imshow(window_name, straightened_card)
            active_windows[window_name] = time.time()

        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
        for (x, y) in box:
            frame = DrawCircle(frame, y, x)

        bottom_y = np.max(box[:, 1]) + 20
        bottom_x = int(np.mean(box[:, 0])) - 50

        cv2.putText(frame, label, (bottom_x, bottom_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 4, lineType=cv2.LINE_AA)

    if len(frame_buffer) == 90:
        save_top_cards()
        frame_buffer.clear()

    current_time = time.time()
    for window_name in list(active_windows):
        if window_name not in current_active_windows:
            if window_name not in inactive_window_times:
                inactive_window_times[window_name] = current_time
            elif current_time - inactive_window_times[window_name] > 0.5:
                cv2.destroyWindow(window_name)
                del active_windows[window_name]
                del inactive_window_times[window_name]
        else:
            inactive_window_times.pop(window_name, None)

    active_windows.update(current_active_windows)

    cv2.imshow('frame', frame)
    #cv2.imshow('mask', mask)

    if cv2.waitKey(1) == ord('q') or cv2.getWindowProperty('frame', cv2.WND_PROP_VISIBLE) < 1:
        cam.release()
        cv2.destroyAllWindows()
        break
