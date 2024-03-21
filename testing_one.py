import tensorflow as tf
import numpy as np
import asl_model

print("Model Loading...")
model = tf.keras.models.load_model('/Users/ltw/Desktop/CS 131 Project/saved_model')
folder_path = '/Users/ltw/Desktop/CS 131 Project/asl_alphabet_data/asl_alphabet_test/'
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
print("Model Loaded...")

path = '/Users/ltw/Desktop/CS 131 Project/asl_alphabet_data/single_tests/a_single_test.jpg'
img = tf.keras.preprocessing.image.load_img(path, target_size=(200, 200))
# matplotlib.pyplot.imshow(img)
img = asl_model.preprocess_image(path)
prediction = model.predict(tf.expand_dims(img, 0))
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
label = 'A'
predicted_label_index = np.argmax(prediction, axis=1)[0]
print(f"Actual: {label}, Predicted: {labels[predicted_label_index]}")