import os
import cv2
from keras.models import load_model
import numpy as np
import time
import pandas as pd
import random

model = load_model('new-model-3.keras')

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

target_word = "VISION"
current_progress = 0
letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

while True:
    _, frame = cap.read()

    h, w, c = frame.shape
    square_side = min(h, w)
    offset = (w - square_side) // 2
    frame = frame[:, offset:offset+square_side]

    for i, letter in enumerate(target_word):
        color = (0, 0, 255) if i >= current_progress else (0, 255, 0)
        cv2.putText(frame, letter, (10 + i*40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    
    k = cv2.waitKey(1)
    if k % 256 == 27: # esc key
        break
    elif k % 256 == 32: # space key
        analysis_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        analysis_frame = cv2.resize(analysis_frame, (28, 28))

        pixel_data = analysis_frame.flatten().reshape(-1, 28, 28, 1) / 255.0
        prediction = model.predict(pixel_data)
        predicted_letter_index = np.argmax(prediction)
        predicted_letter = letterpred[predicted_letter_index]

        print(f"Predicted Character: {predicted_letter}, Confidence: {100*np.max(prediction):.2f}%")
        
        if predicted_letter == target_word[current_progress]:
            current_progress += 1
            print(f"Correct! Progress: {current_progress}/{len(target_word)}")
            if current_progress == len(target_word):
                print("Great Job! You've spelled 'VISION'.")
                # break

    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()
