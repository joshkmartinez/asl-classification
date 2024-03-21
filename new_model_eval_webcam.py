import os
import cv2
from keras.models import load_model
import numpy as np
import time
import pandas as pd

model = load_model('new-model-3.keras')

cap = cv2.VideoCapture(0)

_, frame = cap.read()

h, w, c = frame.shape
square_side = min(h, w)
offset = (w - square_side) // 2

letterpred = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

while True:
    _, frame = cap.read()
    frame = frame[:, offset:offset+square_side]  # Crop to square

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        analysis_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        analysis_frame = cv2.resize(analysis_frame, (28, 28))

        nlist = analysis_frame.flatten().tolist()
        
        datan = pd.DataFrame([nlist])
        datan.columns = list(range(784))

        pixel_data = datan.values / 255
        pixel_data = pixel_data.reshape(-1, 28, 28, 1)
        prediction = model.predict(pixel_data)
        pred_array = np.array(prediction[0])
        letter_prediction_dict = {letterpred[i]: pred_array[i] for i in range(len(letterpred))}
        pred_array_ordered = sorted(pred_array, reverse=True)
        
        high_values = pred_array_ordered[:3]  # Top 3 preds
        for key, value in letter_prediction_dict.items():
            if value in high_values:
                print(f"Predicted Character: {key}, Confidence: {100*value:.2f}%")

        # time.sleep(1)

    cv2.imshow("Frame", frame)

cap.release()
cv2.destroyAllWindows()
