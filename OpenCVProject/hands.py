import cv2
import mediapipe as mp
import math

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Helper function to check if a finger is extended
def is_finger_up(hand_landmarks, finger_tip_idx, finger_dip_idx):
    return hand_landmarks.landmark[finger_tip_idx].y < hand_landmarks.landmark[finger_dip_idx].y

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    gesture_text = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmark positions
            # Indexes: https://google.github.io/mediapipe/solutions/hands#hand-landmark-model
            thumb_up = is_finger_up(hand_landmarks, 4, 3)
            index_up = is_finger_up(hand_landmarks, 8, 6)
            middle_up = is_finger_up(hand_landmarks, 12, 10)
            ring_up = is_finger_up(hand_landmarks, 16, 14)
            pinky_up = is_finger_up(hand_landmarks, 20, 18)

            # Detect gestures
            if thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
                gesture_text = "Thumbs Up 👍"
            elif index_up and middle_up and not ring_up and not pinky_up:
                gesture_text = "Peace ✌"
            elif not thumb_up and not index_up and not middle_up and not ring_up and not pinky_up:
                gesture_text = "Fist ✊"

    # Display gesture
    cv2.putText(frame, gesture_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
