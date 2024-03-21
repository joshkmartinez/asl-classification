import tensorflow as tf
import numpy as np
import os
import asl_model

print("Model Loading...")
model = tf.keras.models.load_model('saved_model/')
print("Model Loaded...")

path_to_data = 'dataset/asl_alphabet_train/asl_alphabet_train'
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']


_, test_dataset = asl_model.create_train_test_datasets(path_to_data, labels, batch_size=32, test_split=0.2)


test_loss, test_acc = model.evaluate(test_dataset)

print('\nTest accuracy:', test_acc)
