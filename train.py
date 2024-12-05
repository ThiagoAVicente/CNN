## TEST SCRIPT WAS COPIED FROM CHAT GPT
# TODO do your own

import numpy as np

from model.cnn import cnn
from functions.load_cifar import *

from functions.save_train import *
from functions.load_model import *

# Load data
data_dir = 'cifar-10-batches-py'  # Specify your CIFAR-10 data directory
selected_classes = [0, 1, 2, 3, 4, 5, 6, 7,8,9]
images, labels = load_cifar10_data(data_dir, num_images_per_class=20, selected_classes=selected_classes)

# Check the shapes of the loaded data
print(f"Images shape: {images.shape}")
print(f"Labels shape: {labels.shape}")

# Define model parameters
input_size = images.shape[1]  # Assuming square images, use 28 for 28x28
num_of_filters = 8
filter_size = 3
num_classes = labels.shape[1]

# Initialize the model
#model = cnn(input_size, num_of_filters, filter_size, num_classes, learning_rate=0.01)

model = load();

# Train the model
model.train(images, labels, limit=1000, batch_size=32)

save(model)
