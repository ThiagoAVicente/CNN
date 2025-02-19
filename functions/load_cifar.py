# :)

import os
import numpy as np
import pickle
from PIL import Image


def load_cifar10_data(data_dir, num_images_per_class=100, selected_classes=None, color = False):
    """
    Loads CIFAR-10 images and labels from the binary data batches, with a limit on classes and images per class.

    :param data_dir: The directory where CIFAR-10 data batches are stored.
    :param num_images_per_class: The number of images to load per class.
    :param selected_classes: List of class indices to include in the dataset.
    :param color: boolean that decides if colored images should also be returned
    :return: Loaded images and one-hot encoded labels.
    """
    images = []
    labels = []
    images_color = []

    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    if selected_classes is None:
        selected_classes = range(7)

    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        with open(batch_file, 'rb') as f:
            batch_data = pickle.load(f, encoding='latin1')
            batch_images = batch_data['data']
            batch_labels = batch_data['labels']

            batch_images = batch_images.reshape((len(batch_labels), 3, 32, 32)).transpose(0, 2, 3, 1)

            # Process images and labels
            for label, image in zip(batch_labels, batch_images):

                # manter só imagens da classe pretendida
                if label in selected_classes:
                    img = Image.fromarray(image)

                    if color:
                        images_color.append(image)

                    img = img.convert('L')  # COnverter para canal unico usando escala cinza
                    img = np.array(img)
                    img = img / 255.0  # Normalizar

                    images.append(img)

                    # Create one-hot encoded label
                    label_vector = np.zeros(len(selected_classes))
                    label_vector[selected_classes.index(label)] = 1
                    labels.append(label_vector)

                # respeitar o número de imagens por classe
                if len(images) >= num_images_per_class * len(selected_classes):
                    break

    images = np.array(images)
    labels = np.array(labels)

    if color:
        images_color = np.array(images_color)
        return images,labels,images_color

    return images, labels
