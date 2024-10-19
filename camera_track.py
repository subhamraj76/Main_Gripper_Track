import cv2
import mediapipe as mp
import math
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while cam.isOpened():
        success, image = cam.read()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
        lmList = []
        if results.multi_hand_landmarks:
            myHand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
        if len(lmList) != 0:
            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]
            cv2.circle(image, (x1, y1), 15, (255, 255, 255))
            cv2.circle(image, (x2, y2), 15, (255, 255, 255))
            cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 3)
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 50:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 3)
            Pos = np.interp(length, [50, 220], [0, 100])
            Posgripper = round(Pos)
            cv2.putText(image, str(Posgripper), (50, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 0))
        cv2.imshow('handDetector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cam.release()
cv2.destroyAllWindows()
