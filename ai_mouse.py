import cv2
import mediapipe as mp
import pyautogui
import time
import math

# Setup kamera 
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# klik & scroll 
click_delay = 0.5
last_left_click = 0
last_right_click = 0
scroll_start_y = None
scroll_sensitivity = 2

# Ukuran layar 
screen_width, screen_height = pyautogui.size()

# Fungsi jarak 
def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# Smoothing
prev_mouse_x, prev_mouse_y = 0, 0
smooth_factor = 0.5

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

        h, w, c = img.shape
        lm = hand.landmark

        # telapak tangan 
        wrist = (int(lm[0].x * w), int(lm[0].y * h))
        index_mcp = (int(lm[5].x * w), int(lm[5].y * h))
        middle_mcp = (int(lm[9].x * w), int(lm[9].y * h))

        palm_x = (wrist[0] + index_mcp[0] + middle_mcp[0]) // 3
        palm_y = (wrist[1] + index_mcp[1] + middle_mcp[1]) // 3

        mouse_x = prev_mouse_x + (palm_x * screen_width / w - prev_mouse_x) * smooth_factor
        mouse_y = prev_mouse_y + (palm_y * screen_height / h - prev_mouse_y) * smooth_factor
        pyautogui.moveTo(mouse_x, mouse_y, duration=0.01)
        prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

        # Koordinat jari 
        thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))
        index_tip = (int(lm[8].x * w), int(lm[8].y * h))
        middle_tip = (int(lm[12].x * w), int(lm[12].y * h))

        # Klik Kiri 
        if distance(thumb_tip, index_tip) < 40:
            if time.time() - last_left_click > click_delay:
                pyautogui.click()
                last_left_click = time.time()
            cv2.circle(img, index_tip, 15, (0,255,0), cv2.FILLED)  # hijau

        # Klik Kanan 
        elif distance(thumb_tip, middle_tip) < 40:
            if time.time() - last_right_click > click_delay:
                pyautogui.rightClick()
                last_right_click = time.time()
            cv2.circle(img, middle_tip, 15, (0,0,255), cv2.FILLED)  # merah

        # Scroll 
        elif distance(index_tip, middle_tip) < 30:
            avg_y = (index_tip[1] + middle_tip[1]) // 2
            if scroll_start_y is None:
                scroll_start_y = avg_y
            scroll_amount = scroll_start_y - avg_y
            if abs(scroll_amount) > 5:
                pyautogui.scroll(scroll_amount * scroll_sensitivity)
                scroll_start_y = avg_y
            cv2.circle(img, ((index_tip[0]+middle_tip[0])//2, (index_tip[1]+middle_tip[1])//2), 15, (255,0,0), cv2.FILLED)  # biru
        else:
            scroll_start_y = None

    cv2.imshow("AI Mouse Gesture (Full)", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
