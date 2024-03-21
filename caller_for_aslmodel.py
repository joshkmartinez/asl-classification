import tensorflow as tf
#from tensorflow.keras import layers, models
import numpy as np
import os
import asl_model

# Define labels and path to data
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']
path_to_data = '/Users/ltw/Desktop/CS 131 Project/asl_alphabet_data/asl_alphabet_train'
train_dataset, test_dataset = asl_model.create_train_test_datasets(path_to_data, labels, batch_size=32, test_split=0.2)
# create_train_test_datasets(root_path, labels, img_height=200, img_width=200, batch_size=32, test_split=0.2):
model = asl_model.create_model(200, 200)
# TODO
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # last layer now a softmax
              metrics=['accuracy'])
trained = model.fit(train_dataset, epochs=2)
# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(test_dataset)

# Print the test accuracy
print('\nTest accuracy:', test_acc)
model.save('/Users/ltw/Desktop/CS 131 Project/saved_model')
print('Model saved to /Users/ltw/Desktop/CS 131 Project/saved_model')
