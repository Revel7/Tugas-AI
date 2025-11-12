import cv2
import mediapipe as mp
import pyautogui
import time
import math
import numpy as np

# = Setup Kamera =
cap = cv2.VideoCapture(0)

# Atur resolusi kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)  # ✋✋ deteksi dua tangan
mp_draw = mp.solutions.drawing_utils

# = Variabel Global =
click_delay = 0.5
last_left_click = 0
last_right_click = 0
scroll_start_y = None
scroll_sensitivity = 2

dragging = False
pinch_start_time = None
pinch_threshold = 0.3
exit_gesture_start = None  # waktu mulai gesture dua tangan

# Ukuran layar
screen_width, screen_height = pyautogui.size()

# Smoothing
prev_mouse_x, prev_mouse_y = 0, 0
smooth_factor = 0.5

# = Fungsi Utilitas =
def distance(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

# = Kalibrasi =
def calibrate_motion_range():
    print("🟡 Kalibrasi dimulai... Gerakkan tanganmu ke segala arah selama 3 detik.")
    start_time = time.time()
    x_positions, y_positions = [], []

    while time.time() - start_time < 3:
        success, img = cap.read()
        img = cv2.flip(img, 1)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            h, w, c = img.shape
            lm = hand.landmark
            wrist = (int(lm[0].x * w), int(lm[0].y * h))
            index_mcp = (int(lm[5].x * w), int(lm[5].y * h))
            middle_mcp = (int(lm[9].x * w), int(lm[9].y * h))
            palm_x = (wrist[0] + index_mcp[0] + middle_mcp[0]) // 3
            palm_y = (wrist[1] + index_mcp[1] + middle_mcp[1]) // 3
            x_positions.append(palm_x)
            y_positions.append(palm_y)

        cv2.putText(img, "Kalibrasi... Gerakkan tanganmu ke seluruh area layar", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("AI Mouse Calibration", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    if not x_positions:
        print("❌ Kalibrasi gagal: tidak ada tangan terdeteksi.")
        return 1.0

    range_x = max(x_positions) - min(x_positions)
    range_y = max(y_positions) - min(y_positions)
    avg_range = (range_x + range_y) / 2

    base_range = 250
    sensitivity = base_range / avg_range
    sensitivity = np.clip(sensitivity, 0.5, 3.0)

    print(f"✅ Kalibrasi selesai. Sensitivitas: {sensitivity:.2f}")
    cv2.destroyWindow("AI Mouse Calibration")
    return sensitivity

# kalibrasi awal
sensitivity = calibrate_motion_range()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    open_hands = 0  # jumlah tangan yang terbuka penuh

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            h, w, c = img.shape
            lm = hand_landmarks.landmark

            wrist = (int(lm[0].x * w), int(lm[0].y * h))
            index_tip = (int(lm[8].x * w), int(lm[8].y * h))
            middle_tip = (int(lm[12].x * w), int(lm[12].y * h))
            ring_tip = (int(lm[16].x * w), int(lm[16].y * h))
            pinky_tip = (int(lm[20].x * w), int(lm[20].y * h))
            thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))

            index_mcp = (int(lm[5].x * w), int(lm[5].y * h))
            middle_mcp = (int(lm[9].x * w), int(lm[9].y * h))
            palm_x = (wrist[0] + index_mcp[0] + middle_mcp[0]) // 3
            palm_y = (wrist[1] + index_mcp[1] + middle_mcp[1]) // 3

            # Hitung jari terbuka
            open_fingers = 0
            for tip, mcp_id in zip([4, 8, 12, 16, 20], [2, 5, 9, 13, 17]):
                if lm[tip].y < lm[mcp_id].y:
                    open_fingers += 1
            if open_fingers >= 5:
                open_hands += 1  # tangan ini terbuka penuh

            # tangan kanan untuk kontrol mouse
            if result.multi_hand_landmarks.index(hand_landmarks) == 0:
                mouse_x = prev_mouse_x + ((palm_x * screen_width / w - prev_mouse_x) * smooth_factor * sensitivity)
                mouse_y = prev_mouse_y + ((palm_y * screen_height / h - prev_mouse_y) * smooth_factor * sensitivity)
                pyautogui.moveTo(mouse_x, mouse_y, duration=0.01)
                prev_mouse_x, prev_mouse_y = mouse_x, mouse_y

                dist_thumb_index = distance(thumb_tip, index_tip)
                dist_thumb_middle = distance(thumb_tip, middle_tip)
                dist_index_middle = distance(index_tip, middle_tip)

                # = Klik & Drag =
                if dist_thumb_index < 35:
                    if pinch_start_time is None:
                        pinch_start_time = time.time()
                    hold_time = time.time() - pinch_start_time
                    if hold_time > pinch_threshold:
                        if not dragging:
                            pyautogui.mouseDown()
                            dragging = True
                        cv2.circle(img, index_tip, 15, (0, 255, 255), cv2.FILLED)
                else:
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                    elif pinch_start_time is not None:
                        if (time.time() - pinch_start_time) <= pinch_threshold:
                            if time.time() - last_left_click > click_delay:
                                pyautogui.click()
                                last_left_click = time.time()
                        pinch_start_time = None

                # = Klik kanan =
                if dist_thumb_middle < 40:
                    if time.time() - last_right_click > click_delay:
                        pyautogui.rightClick()
                        last_right_click = time.time()
                    cv2.circle(img, middle_tip, 15, (0, 0, 255), cv2.FILLED)

                # = Scroll =
                elif dist_index_middle < 30:
                    avg_y = (index_tip[1] + middle_tip[1]) // 2
                    if scroll_start_y is None:
                        scroll_start_y = avg_y
                    scroll_amount = scroll_start_y - avg_y
                    if abs(scroll_amount) > 5:
                        pyautogui.scroll(scroll_amount * scroll_sensitivity)
                        scroll_start_y = avg_y
                    cv2.circle(img, ((index_tip[0]+middle_tip[0])//2, (index_tip[1]+middle_tip[1])//2),
                               15, (255, 0, 0), cv2.FILLED)
                else:
                    scroll_start_y = None

    if open_hands >= 2:
        if exit_gesture_start is None:
            exit_gesture_start = time.time()
        elif time.time() - exit_gesture_start > 2:
            print("👋 Dua tangan terbuka terdeteksi! Menutup program...")
            break
        cv2.putText(img, "Tahan dua tangan terbuka 2 detik untuk keluar", (40, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        exit_gesture_start = None

    # = Tampilan Kamera =
    scale = 1.5
    new_width = int(img.shape[1] * scale)
    new_height = int(img.shape[0] * scale)
    resized = cv2.resize(img, (new_width, new_height))

    cv2.putText(resized, f"Sensitivity: {sensitivity:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(resized, "Tekan ESC / Dua Tangan Terbuka 2 Detik untuk Keluar", (20, new_height - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

    cv2.imshow("AI Mouse (2-Hand Exit + Zoom)", resized)

    # = Keluar pakai ESC =
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
