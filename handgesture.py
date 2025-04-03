import cv2
import mediapipe as mp
import time
import numpy as np
import os
from urllib.request import urlretrieve

# Updated gesture image URLs
ICON_URLS = {
    0: "https://w7.pngwing.com/pngs/224/438/png-transparent-caves-of-gargas-hand-free-content-cartoon-hands-s-face-hand-head-thumbnail.png",  # New Palm image
    1: "https://images.vexels.com/media/users/3/265103/isolated/preview/e8896baa8390e6402c97b820a08b442c-hand-doing-peace-sign.png",  # Peace Sign
    2: "https://cdn-icons-png.flaticon.com/512/7521/7521996.png"   # Middle Three
}

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define gesture names
GESTURES = {
    0: "Open Palm",
    1: "Peace Sign",
    2: "Middle Three"
}

# Game variables
sequence = [0, 2, 1]  # Palm, Middle Three, Peace
current_sequence_index = 0
attempts = 0
TOTAL_ATTEMPTS = 6  # Changed from 15 to 6
points = 0
lives = 3
last_gesture = None
gesture_start_time = None
gesture_confirmed = False
GESTURE_CONFIRMATION_DELAY = 1.0
showing_sequence = True

# Load gesture icons
gesture_icons = {}
for gesture_id, url in ICON_URLS.items():
    try:
        filename = f"gesture_{gesture_id}.png"
        if not os.path.exists(filename):
            urlretrieve(url, filename)
        icon = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if icon is None:
            raise ValueError("Failed to load image")
        
        # Convert to BGRA if needed
        if icon.shape[2] == 3:
            icon = cv2.cvtColor(icon, cv2.COLOR_BGR2BGRA)
        gesture_icons[gesture_id] = cv2.resize(icon, (150, 150))
    except Exception as e:
        print(f"Error loading icon for gesture {gesture_id}: {e}")
        # Fallback to colored rectangle
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        gesture_icons[gesture_id] = np.zeros((150, 150, 4), dtype=np.uint8)
        gesture_icons[gesture_id][:, :, :3] = colors[gesture_id]
        gesture_icons[gesture_id][:, :, 3] = 255  # Full opacity

def detect_gesture(landmarks):
    if not landmarks:
        return None
    
    thumb_up = landmarks[4].y < landmarks[3].y
    index_up = landmarks[8].y < landmarks[6].y
    middle_up = landmarks[12].y < landmarks[10].y
    ring_up = landmarks[16].y < landmarks[14].y
    pinky_up = landmarks[20].y < landmarks[18].y

    if index_up and middle_up and ring_up and pinky_up:
        return 0  # Open Palm
    if thumb_up and index_up and middle_up and not ring_up and not pinky_up:
        return 1  # Peace Sign
    if thumb_up and index_up and middle_up and ring_up and not pinky_up:
        return 2  # Middle Three
    return None

def overlay_transparent(background, overlay, x, y):
    h, w = overlay.shape[:2]
    
    # Extract alpha channel
    if overlay.shape[2] == 4:
        alpha = overlay[:, :, 3] / 255.0
        overlay = overlay[:, :, :3]
    else:
        alpha = np.ones(overlay.shape[:2], dtype=np.float32)
    
    # Region of interest
    roi = background[y:y+h, x:x+w]
    
    # Blend images
    for c in range(3):
        roi[:, :, c] = (1 - alpha) * roi[:, :, c] + alpha * overlay[:, :, c]
    
    return background

# Main game loop
while lives > 0 and attempts < TOTAL_ATTEMPTS:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    height, width = frame.shape[:2]
    
    # Process frame with MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    
    # Display game info
    cv2.putText(frame, f"Round: {attempts+1}/{TOTAL_ATTEMPTS}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Points: {points}  Lives: {lives}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Show current gesture or wait for player
    current_gesture = sequence[attempts % len(sequence)]
    if showing_sequence:
        cv2.putText(frame, "Simon Says:", (width//2 - 100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # Overlay gesture icon
        icon = gesture_icons[current_gesture]
        frame = overlay_transparent(frame, icon, width//2 - 75, 100)
        
        cv2.imshow("Gesture Simon Says", frame)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
            
        showing_sequence = False
    else:
        # Player input
        detected_gesture = None
        if results.multi_hand_landmarks:
            detected_gesture = detect_gesture(results.multi_hand_landmarks[0].landmark)
        
        # Display detected gesture
        if detected_gesture is not None:
            cv2.putText(frame, f"Detected: {GESTURES[detected_gesture]}", 
                        (width//2 - 120, height - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Gesture confirmation
        current_time = time.time()
        if detected_gesture is not None:
            if detected_gesture != last_gesture:
                last_gesture = detected_gesture
                gesture_start_time = current_time
                gesture_confirmed = False
            else:
                if not gesture_confirmed and (current_time - gesture_start_time) >= GESTURE_CONFIRMATION_DELAY:
                    if detected_gesture == current_gesture:
                        attempts += 1
                        points += 5
                        gesture_confirmed = True
                        cv2.putText(frame, "Correct! +5 Points", (width//2 - 120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        lives -= 1
                        cv2.putText(frame, f"Wrong! Lives left: {lives}", (width//2 - 120, 200),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    cv2.imshow("Gesture Simon Says", frame)
                    cv2.waitKey(1500)
                    showing_sequence = True
                    last_gesture = None
    
    cv2.imshow("Gesture Simon Says", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Game over screen
frame = np.zeros((480, 640, 3), dtype=np.uint8)
if attempts >= TOTAL_ATTEMPTS:
    cv2.putText(frame, "YOU WON!", (width//2 - 100, height//2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
else:
    cv2.putText(frame, "GAME OVER", (width//2 - 150, height//2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
cv2.putText(frame, f"Final Score: {points}", (width//2 - 120, height//2 + 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.imshow("Gesture Simon Says", frame)
cv2.waitKey(3000)

# Cleanup
cap.release()
cv2.destroyAllWindows()