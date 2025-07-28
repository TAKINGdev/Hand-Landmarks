import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands= 2, min_detection_confidence= 0.7)
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    frame_height, frame_width = frame.shape[:2]
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    black_canvas = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(black_canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS)


    cv2.imshow("Main Camera", frame)
    cv2.imshow("Detected Hands Landmarks", black_canvas)

    key = cv2.waitKey(1)
    if key == ord('q') :
        break

cap.release()
cv2.destroyAllWindows()