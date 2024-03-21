import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

def preprocess_image(path, img_height=200, img_width=200):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3) # edit to use grayscale
    img = tf.image.resize(img, [img_height, img_width])
    img /= 255.0 # normalization for tensorfl0w
    return img

def create_train_test_datasets(root_path, labels, img_height=200, img_width=200, batch_size=32, test_split=0.2):
    print('Creating datasets...')
    image_paths = []
    image_labels = []
    label_to_index = {label: index for index, label in enumerate(labels)}
    
    for label in labels:
        label_path = os.path.join(root_path, label)
        for img_name in os.listdir(label_path):
            img_path = os.path.join(label_path, img_name)
            image_paths.append(img_path)
            image_labels.append(label_to_index[label])
    
    # Shuffle the dataset
    rng = np.random.default_rng()
    indices = np.arange(len(image_paths))
    rng.shuffle(indices)
    image_paths = np.array(image_paths)[indices]
    image_labels = np.array(image_labels)[indices]
    

    image_paths = tf.convert_to_tensor(image_paths)
    image_labels = tf.convert_to_tensor(image_labels)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, image_labels))
    dataset = dataset.map(lambda x, y: (preprocess_image(x, img_height, img_width), y),
                          num_parallel_calls=tf.data.AUTOTUNE)
    
    total_batches = len(image_paths) // batch_size
    test_batches = int(total_batches * test_split)
    train_batches = total_batches - test_batches
    
    train_dataset = dataset.skip(test_batches).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    test_dataset = dataset.take(test_batches).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_dataset, test_dataset

def create_model(img_height=200, img_width=200):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(), 
        layers.Dense(128, activation='relu'),
        layers.Dense(len(labels))
    ])
    return model
