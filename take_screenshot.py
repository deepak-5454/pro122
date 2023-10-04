import numpy as np
import pyautogui

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips = [8, 12, 16, 20]
thumb_tip = 4

palm_closed = False  # Flag to track if the palm is closed

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            # accessing the landmarks by their position
            lm_list = []
            for id, lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # array to hold true or false if finger is folded
            finger_fold_status = []
            for tip in finger_tips:
                # getting the landmark tip position and drawing a blue circle
                x, y = int(lm_list[tip].x * w), int(lm_list[tip].y * h)
                cv2.circle(img, (x, y), 15, (255, 0, 0), cv2.FILLED)

                # writing a condition to check if finger is folded
                if lm_list[tip].x < lm_list[tip - 3].x:
                    cv2.circle(img, (x, y), 15, (0, 255, 0), cv2.FILLED)
                    finger_fold_status.append(True)
                else:
                    finger_fold_status.append(False)

            # Checking if all fingers are folded
            if all(finger_fold_status):
                palm_closed = True
            else:
                palm_closed = False

            mp_draw.draw_landmarks(img, hand_landmark,
                                   mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0, 0, 255), 2, 2),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

    cv2.imshow("hand tracking", img)

    # Capture a screenshot when the palm is closed
    if palm_closed:
        screenshot = pyautogui.screenshot()
        screenshot.save("screenshot.png")  # Save the screenshot to a file

    if cv2.waitKey(1) == 27:  # Press 'Esc' key to exit
        break

cap.release()
cv2.destroyAllWindows()
