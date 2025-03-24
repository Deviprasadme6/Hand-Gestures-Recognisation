import cv2
import mediapipe as mp

# Define MediaPipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Define function to recognize hand gestures
def recognize_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Gesture 'OK' - Thumb and index finger form a circle
    if abs(thumb_tip.x - index_tip.x) < 0.05 and abs(thumb_tip.y - index_tip.y) < 0.05:
        return 'OK'

    # Gesture 'Hello' - Fingers spread out with thumb above other fingers
    if index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y and thumb_tip.y < index_tip.y:
        return 'Hello'

    # Gesture 'Thank You' - Thumb is extended and other fingers are bent or relaxed
    if thumb_tip.y < index_tip.y and abs(thumb_tip.x - index_tip.x) > 0.1:
        return 'Thank You'

    # Gesture 'Please' - Palm open, with fingers spread out
    if thumb_tip.y < index_tip.y and index_tip.y < middle_tip.y < ring_tip.y < pinky_tip.y:
        return 'Please'

    # Gesture 'Stop' - Hand with fingers together and palm facing outward
    if thumb_tip.x < index_tip.x and index_tip.x < middle_tip.x and middle_tip.x < ring_tip.x and ring_tip.x < pinky_tip.x:
        return 'Stop'

    # Gesture 'Yes' - Thumb up
    if thumb_tip.y < index_tip.y and thumb_tip.x > index_tip.x:
        return 'Yes'

    # Gesture 'No' - Index finger pointing left or right
    if index_tip.x < middle_tip.x and index_tip.y < middle_tip.y:
        return 'No'

    return 'Unknown Gesture'

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert frame to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and get hand landmarks
    results = hands.process(rgb_frame)

    # Draw landmarks and recognize gesture
    if results.multi_hand_landmarks:
        for landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = recognize_gesture(landmarks.landmark)
            cv2.putText(frame, f"Gesture: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the frame with the hand gestures
    cv2.imshow('Hand Gesture Recognition', frame)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
