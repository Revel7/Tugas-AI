import cv2
import mediapipe as mp
import pyautogui
import time

pyautogui.FAILSAFE = False

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, 
                       min_detection_confidence=0.40, 
                       min_tracking_confidence=0.30)
mp_draw = mp.solutions.drawing_utils

screen_w, screen_h = pyautogui.size()

smooth = 0.25
prev_x, prev_y = 0, 0

fps_limit = 25
prev_time = 0

# var click & scroll
touch_start = 0
is_dragging = False
hold_time = 0.20

drag_threshold = 65
click_threshold = 45

double_click_interval = 0.5
last_click_time = 0
left_click_locked = False

last_right_click_time = 0
right_click_delay = 0.5
scroll_start_y = None

while True:
    ret, cam = cap.read()
    if not ret:
        break

    current = time.time()
    if (current - prev_time) < (1 / fps_limit):
        continue
    prev_time = current

    cam = cv2.flip(cam, 1)
    rgb = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    h, w, c = cam.shape
    cam_center_x = w // 2
    cam_center_y = h // 2

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(cam, hand, mp_hands.HAND_CONNECTIONS)

        lm = hand.landmark

        wrist = (int(lm[0].x * w), int(lm[0].y * h))
        index_mcp = (int(lm[5].x * w), int(lm[5].y * h))
        middle_mcp = (int(lm[9].x * w), int(lm[9].y * h))

        palm_x = (wrist[0] + index_mcp[0] + middle_mcp[0]) // 3
        palm_y = (wrist[1] + index_mcp[1] + middle_mcp[1]) // 3

        # posisi jari
        index_tip = (int(lm[8].x * w), int(lm[8].y * h))
        thumb_tip = (int(lm[4].x * w), int(lm[4].y * h))
        thumb_mid = (
            int((lm[4].x + lm[3].x ) / 2 * w),
            int((lm[4].y + lm[3].y ) / 2 * h)
        )
        middle_tip = (int(lm[12].x * w), int(lm[12].y * h))

        # Gerakan Mouse
        dx = palm_x - cam_center_x
        dy = palm_y - cam_center_y

        # Skala agar kamera = mouse pad
        sesitivity = 1.8
        scale_x = (screen_w / w) * sesitivity
        scale_y = (screen_h / h) * sesitivity

        target_x = (screen_w / 2) + (dx * scale_x)
        target_y = (screen_h / 2) + (dy * scale_y)

        # Smooth
        mouse_x = prev_x + (target_x - prev_x) * smooth
        mouse_y = prev_y + (target_y - prev_y) * smooth

        pyautogui.moveTo(mouse_x, mouse_y)
        prev_x, prev_y = mouse_x, mouse_y

        # fungsi drag
        dist_drag = ((index_tip[0] - thumb_mid[0]) ** 2 + (index_tip[1] - thumb_mid[1]) ** 2) ** 0.5
        now = time.time()

        if dist_drag < drag_threshold and not is_dragging:
            if touch_start == 0:
                touch_start = now

            if now - touch_start >= hold_time:
                pyautogui.mouseDown()
                is_dragging = True
                cv2.putText(cam, "Dragging", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.circle(cam, index_tip, 15, (255, 0, 255), cv2.FILLED)
        elif dist_drag >= drag_threshold:
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
            touch_start = 0 

        # Fungsi klik
        dist = ((index_tip[0] - thumb_tip[0]) ** 2 + (index_tip[1] - thumb_tip[1]) ** 2) ** 0.5

        # if abs(dist - prev_dist) < 10:
        #     dist = prev_dist
        
        # prev_dist = dist

        if dist < click_threshold:
                now = time.time()

                # Double click
                if now - last_click_time <= double_click_interval:
                    pyautogui.doubleClick()
                    last_click_time = 0  # reset
                    time.sleep(0.2)
                    cv2.putText(cam, "Double Click", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.circle(cam, index_tip, 15, (0,0,255), cv2.FILLED)
                # Klik kiri
                else:
                    pyautogui.click()
                    last_click_time = now
                    time.sleep(0.2)
                    cv2.putText(cam, "Click", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    cv2.circle(cam, index_tip, 15, (0,255,0), cv2.FILLED)
        # Klik kanan
        dist_right = ((middle_tip[0] - thumb_tip[0]) ** 2 + (middle_tip[1] - thumb_tip[1]) ** 2) ** 0.5
        
        if dist_right < click_threshold + 5:
            now = time.time()
            if now - last_right_click_time > right_click_delay:
                pyautogui.rightClick()
                last_right_click_time = now
                time.sleep(0.25)
                cv2.putText(cam, "Right click", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.circle(cam, middle_tip, 15, (0, 0, 255), cv2.FILLED)

        # scroll
        dist_scroll = ((index_tip[0] - middle_tip[0]) ** 2 + (index_tip[1] - middle_tip[1]) ** 2) ** 0.5
        
        
        if dist_scroll < click_threshold - 5:
            h,w, _ = cam.shape
            frame_mid = h // 2

            hand_y = (index_tip[1] + middle_tip[1]) // 2

            dead_zone = 40
            scroll_speed = 80

            if hand_y < frame_mid - dead_zone:
                pyautogui.scroll(scroll_speed)
                cv2.putText(cam, "Scroll Up", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.circle(cam, ((index_tip[0] + middle_tip[0]) // 2, (index_tip[1] + middle_tip[1]) // 2),
                           15, (0, 255, 255), cv2.FILLED)
                
            elif hand_y > frame_mid + dead_zone:
                pyautogui.scroll(-scroll_speed)
                cv2.putText(cam, "Scroll Down", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.circle(cam, ((index_tip[0] + middle_tip[0]) // 2, (index_tip[1] + middle_tip[1]) // 2),
                           15, (0, 255, 255), cv2.FILLED)

    cv2.imshow("kamera", cam)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
