import network
from mnist_loader import load_data_wrapper

def main():
    # Load the data using the mnist_loader
    training_data, validation_data, test_data = load_data_wrapper()

    # Create a network instance
    net = network.Network([784, 30, 10])

    # Train the network
    net.SGD(training_data, epochs=5, mini_batch_size=10, eta=3.0, test_data=test_data)

if __name__ == "__main__":
    main()

