import cv2
import mediapipe as mp
import pytermgui as ptg

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Inisialisasi VideoCapture dan MediaPipe Hands
cap = cv2.VideoCapture(0)
hands = mp_hands.Hands()

# Ukuran terminal
terminal_width = 80  # Sesuaikan dengan ukuran terminal
terminal_height = 24

try:
    while True:
        data, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Mendapatkan koordinat ujung jari telunjuk
                finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                h, w, _ = image.shape
                x, y = int(finger_tip.x * w), int(finger_tip.y * h)

                # Konversi koordinat piksel ke skala terminal
                term_x = int((x / w) * terminal_width)
                term_y = int((y / h) * terminal_height)

                # Gerakkan kursor di terminal
                ptg.move_cursor(term_x, term_y)
                print("*", end="", flush=True)

                # Menampilkan landmark di video
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Handtracker", image)
        if cv2.waitKey(1) & 0xFF == 27:  # Tekan ESC untuk keluar
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
