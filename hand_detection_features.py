import cv2
import numpy as np

def main():
    orb = cv2.ORB_create()

    template = cv2.imread('dataset/asl_alphabet_test/asl_alphabet_test/A_test.jpg', 0)
    if template is None:
        print("Error loading template")
        return
    
    kp_template, des_template = orb.detectAndCompute(template, None)

    if des_template is None:
        print("No descriptors found in template.")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video stream or file")
        return

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)

            if des_frame is None:
                print("No descriptors found in frame.")
                continue

            if des_template.dtype != des_frame.dtype:
                des_frame = des_frame.astype(des_template.dtype)

            matches = bf.match(des_template, des_frame)
            matches = sorted(matches, key=lambda x: x.distance)

            matched_frame = cv2.drawMatches(template, kp_template, frame, kp_frame, matches[:10], None, flags=2)

            cv2.imshow('Hand Detection', matched_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
