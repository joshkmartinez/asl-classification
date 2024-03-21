import cv2

template = cv2.imread('dataset/asl_alphabet_test/asl_alphabet_test/A_test.jpg', 0)
w, h = template.shape[::-1]

similarity_threshold = 0.65

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    result = cv2.matchTemplate(gray_frame, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val >= similarity_threshold:
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv2.rectangle(frame, top_left, bottom_right, (255, 0, 0), 2)
    else:
        print("No hand detected in the frame with high enough confidence.")

    cv2.imshow('Hand Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
