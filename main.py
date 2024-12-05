import pickle
from model.cnn import *
from functions.load_model import *
from functions.load_cifar import *
import random

model = load();

data_dir = 'cifar-10-batches-py'
selected_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
images, labels = load_cifar10_data(data_dir, num_images_per_class=20, selected_classes=selected_classes)

indices = random.sample(range(images.shape[0]), 100)
selected_images = images[indices]
selected_labels = labels[indices]

correct_predictions = 0

for i in range(len(selected_images)):
    image = selected_images[i]
    label = selected_labels[i]
    prediction = model.forward_propagation(image)

    if np.argmax(prediction) == np.argmax(label):
        correct_predictions += 1

accuracy = correct_predictions / 100
print(f"Accuracy: {accuracy * 100:.2f}%")
