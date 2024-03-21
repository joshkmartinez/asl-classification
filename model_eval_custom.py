import tensorflow as tf
import numpy as np
import os
import glob
import asl_model

model_path = 'saved_model/'
model = tf.keras.models.load_model(model_path)

image_directory = 'preprocessed_temp/'
image_paths = glob.glob(os.path.join(image_directory, '*.jpg'))  

for image_path in image_paths:
    img = asl_model.preprocess_image(image_path)
    
    img_batch = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img_batch)
    predicted_label_index = np.argmax(prediction, axis=1)[0]
    
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
    predicted_label = labels[predicted_label_index]
    
    print(f"Image: {os.path.basename(image_path)}, Prediction: {predicted_label}")
