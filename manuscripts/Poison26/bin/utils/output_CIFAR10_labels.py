import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

def standardize_cifar(x):
    """Standardize CIFAR10 data to the [0, 1] range."""
    return x / 255.0

def load_data(batch_size=32):
    """
    Loads CIFAR10 train and test data and prepares it for evaluation.

    Args:
        batch_size (int): The batch size for data loading.

    Returns:
        tf.data.Dataset, tf.data.Dataset: The training and testing datasets.
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Reshape and normalize the data
    x_train = standardize_cifar(x_train.astype('float32'))
    x_test = standardize_cifar(x_test.astype('float32'))

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create TensorFlow datasets and batch them
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_dataset, test_dataset

if __name__ == '__main__':
    # Set the batch size and load the datasets
    batch_size = 32
    train_dataset, test_dataset = load_data(batch_size=batch_size)

    # Write the training labels and their associated index to a TSV file.
    with open('CIFAR10_train_labels.tsv', 'w') as train_file:
        # Optional: write a header line
        train_file.write("Index\tLabel\n")
        # Unbatch the dataset to iterate over individual examples.
        for idx, (_, label) in enumerate(train_dataset.unbatch()):
            # Convert one-hot encoded label back to an integer.
            true_label = int(np.argmax(label.numpy()))
            train_file.write(f"{idx}\t{true_label}\n")

    # Write the testing labels and their associated index to a separate TSV file.
    with open('CIFAR10_test_labels.tsv', 'w') as test_file:
        # Optional: write a header line
        test_file.write("Index\tLabel\n")
        for idx, (_, label) in enumerate(test_dataset.unbatch()):
            true_label = int(np.argmax(label.numpy()))
            test_file.write(f"{idx}\t{true_label}\n")

    print("Labels with their associated indexes have been written to 'CIFAR10_train_labels.tsv' and 'CIFAR10_test_labels.tsv'.")

