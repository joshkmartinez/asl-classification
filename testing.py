import tensorflow as tf
import numpy as np
import asl_model

print("Model Loading...")
model = tf.keras.models.load_model('saved_model/')
folder_path = 'dataset/asl_alphabet_test/asl_alphabet_test/'
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
print("Model Loaded...")
### Dont modify above this line

# for label in labels:
#     path = folder_path + label + '_test.jpg'
#     img = asl_model.preprocess_image(path)
#     print(label)
#     prediction = model.predict(tf.expand_dims(img, 0))


incorrect_predictions = 0  # Initialize counter

for label in labels:
    if(label == "del"):
        continue
    
    path = folder_path + label + '_test.jpg'
    img = asl_model.preprocess_image(path)
    prediction = model.predict(tf.expand_dims(img, 0))
    predicted_label_index = np.argmax(prediction, axis=1)[0]  # Get the index of the max prediction score
    
    # Assuming your labels list is in the same order as the model's output,
    # this will check if the prediction matches the actual label
    if labels[predicted_label_index] != label:
        incorrect_predictions += 1
    print(f"Actual: {label}, Predicted: {labels[predicted_label_index]}")

# After the loop
print(f"Total incorrect predictions: {incorrect_predictions}")
print(f"Accuracy: {(len(labels) - incorrect_predictions) / len(labels) * 100:.2f}%")

 