import cv2
import mediapipe as mp

def main():
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                coords = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
                x_coords = [coord[0] for coord in coords]
                y_coords = [coord[1] for coord in coords]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                width, height = x_max - x_min, y_max - y_min
                box_start = (int(x_min * image.shape[1]), int(y_min * image.shape[0]))
                box_end = (int(x_max * image.shape[1]), int(y_max * image.shape[0]))
                cv2.rectangle(image, box_start, box_end, (0, 255, 0), 2)

        cv2.imshow('MediaPipe Hands', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    hands.close()
    cap.release()

if __name__ == "__main__":
    main()
