import cv2
import mediapipe as mp
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_drawing_style = mp.solutions.drawing_styles
mphands = mp.solutions.hands

cap = cv2.VideoCapture(1)
hands = mphands.Hands()

def is_hand_open(hand_landmarks):
    # Landmark jari
    finger_tips = [
        mphands.HandLandmark.THUMB_TIP,
        mphands.HandLandmark.INDEX_FINGER_TIP,
        mphands.HandLandmark.MIDDLE_FINGER_TIP,
        mphands.HandLandmark.RING_FINGER_TIP,
        mphands.HandLandmark.PINKY_TIP,
    ]
    finger_dips = [
        mphands.HandLandmark.THUMB_IP,
        mphands.HandLandmark.INDEX_FINGER_PIP,
        mphands.HandLandmark.MIDDLE_FINGER_PIP,
        mphands.HandLandmark.RING_FINGER_PIP,
        mphands.HandLandmark.PINKY_PIP,
    ]
    # Periksa kalo ujung jari lebih tinggi dari yang deket
    return all(
        hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y
        for tip, dip in zip(finger_tips, finger_dips)
    )

while True:
    data, image = cap.read()
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Untuk kordinat jari telunjuk
            finger_tip = hand_landmarks.landmark[mphands.HandLandmark.INDEX_FINGER_TIP]
            h, w, _ = image.shape  # Mendapatkan tinggi dan lebar frame
            x, y = int(finger_tip.x * w), int(finger_tip.y * h)  # Konversi ke piksel

            print(f"Ujung jari telunjuk: x={x}, y={y}")
            pyautogui.moveTo(x, y, duration=0.1)

            cv2.putText(image, f"x: {x}, y: {y}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Periksa apakah semua jari dalam posisi terbuka (5 jari)
            if is_hand_open(hand_landmarks):
                pyautogui.click()
                print("Klik dilakukan!")

            # Menggambar landmark dan koneksi
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mphands.HAND_CONNECTIONS
            )
    
    cv2.imshow('Handtracker', image)
    if cv2.waitKey(1) & 0xFF == 27:  
        break

cap.release()
cv2.destroyAllWindows()
