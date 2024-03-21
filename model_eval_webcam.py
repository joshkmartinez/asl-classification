import cv2
import tensorflow as tf
import numpy as np
import asl_model
import os

model_path = 'saved_model/'
model = tf.keras.models.load_model(model_path)

cap = cv2.VideoCapture(0)

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

def preprocess_image_webcam(image, target_height=200, target_width=200, channels=3, save_path=None):
    target_aspect = target_width / target_height
    
    (h, w) = image.shape[:2]
    current_aspect = w / h

    if current_aspect > target_aspect:
        new_width = int(target_aspect * h)
        offset = (w - new_width) // 2
        image = image[:, offset:offset+new_width]
    elif current_aspect < target_aspect:
        new_height = int(w / target_aspect)
        offset = (h - new_height) // 2
        image = image[offset:offset+new_height, :]
    
    if channels == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    image = cv2.resize(image, (target_width, target_height))
    
    image = np.array(image, dtype=np.float32) / 255.0

    if save_path is not None:
        if channels == 3:
            image_to_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_to_save = image
        cv2.imwrite(save_path, image_to_save * 255) 
    
    
    return image

frame_count=0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    path = f"preprocessed_temp/frame_{frame_count}.jpg"
    frame_count += 1
    save_dir = path.split("/")[0]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    img = preprocess_image_webcam(frame, save_path=path)

    prediction = model.predict(tf.expand_dims(img, 0))
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    predicted_label = labels[predicted_label_index]
    cv2.putText(frame, f'Prediction: {predicted_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
