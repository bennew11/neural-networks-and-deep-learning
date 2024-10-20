import matplotlib.pyplot as plt
import numpy as np
from mnist_loader import load_data

def show_image(image):
    # Reshape the 784-dimensional vector back into a 28x28 image
    image = np.reshape(image, (28, 28))
    plt.imshow(image, cmap='gray')
    plt.show()

def inspect_data():
    training_data, validation_data, test_data = load_data()

    train_inputs, train_labels = training_data

    # Show the first image and its label
    print("Label of the first training image:", train_labels[1])
    show_image(train_inputs[1])

"""if __name__ == "__main__":
    inspect_data()"""
