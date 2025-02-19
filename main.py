import pickle
from model.cnn import *
from functions.load_model import *
from functions.load_cifar import *
import random
import matplotlib.pyplot as plt

model = load();

data_dir = 'cifar-10-batches-py'
classes2Txt = {0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}
selected_classes = [x for x in range(10)]
images, labels, colored = load_cifar10_data(data_dir, num_images_per_class=20, selected_classes=selected_classes, color = True)

indices = random.sample(range(images.shape[0]), 100)
selected_images = images[indices]
selected_labels = labels[indices]
selected_colored = colored[indices]

correct_predictions = 0

for i in range(len(selected_images)):
    image = selected_images[i]
    label = selected_labels[i]
    prediction = model.forward_propagation(image)

    predicted_class = np.argmax(prediction)
    true_class = np.argmax(label)


    if predicted_class == true_class:
        correct_predictions += 1

    plt.figure(figsize=(10, 5))
    plt.imshow(selected_colored[i],interpolation='nearest')
    plt.title(f"Predicted: {classes2Txt[int(predicted_class)]}, True: {classes2Txt[int(true_class)]}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

accuracy = correct_predictions / 100
print(f"Accuracy: {accuracy * 100:.2f}%")
