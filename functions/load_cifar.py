## FUNCTION WAS COPIED FROM CHAT GPT

import os
import numpy as np
import pickle
from PIL import Image


def load_cifar10_data(data_dir, num_images_per_class=100, selected_classes=None):
    """
    Loads CIFAR-10 images and labels from the binary data batches, with a limit on classes and images per class.

    :param data_dir: The directory where CIFAR-10 data batches are stored.
    :param num_images_per_class: The number of images to load per class.
    :param selected_classes: List of class indices to include in the dataset.
    :return: Loaded images and one-hot encoded labels.
    """
    images = []
    labels = []

    # Define the CIFAR-10 class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if selected_classes is None:
        selected_classes = range(7)  # Use the first 7 classes

    # Loop through all data batch files
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f, encoding='latin1')
            batch_images = batch_data['data']
            batch_labels = batch_data['labels']

            # Convert each image from 3072 to 3x32x32
            batch_images = batch_images.reshape((len(batch_labels), 3, 32, 32)).transpose(0, 2, 3, 1)

            # Process images and labels
            for label, image in zip(batch_labels, batch_images):
                if label in selected_classes:  # Only keep images for the selected classes
                    img = Image.fromarray(image)
                    img = img.convert('L')  # Convert to grayscale (single channel)
                    img = img.resize((28, 28))  # Resize to 28x28 for compatibility with the model
                    img = np.array(img)
                    img = img / 255.0  # Normalize image to [0, 1]

                    images.append(img)
                    # Create one-hot encoded label
                    label_vector = np.zeros(len(selected_classes))
                    label_vector[selected_classes.index(label)] = 1
                    labels.append(label_vector)

                # Limit total number of images to 700
                if len(images) >= num_images_per_class * len(selected_classes):
                    break

    images = np.array(images)
    labels = np.array(labels)
    return images, labels
