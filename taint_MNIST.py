import argparse
import time
import numpy as np
from tqdm import tqdm
<<<<<<< HEAD
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.models import load_model  # Use `load_model` from Keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt
=======
>>>>>>> 2e3720e3acf18d382025057cfe30b34846348776
import tensorflow as tf
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.datasets import mnist, cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from Adversarial_Observation.utils import seed_everything
from Adversarial_Observation import AdversarialTester, ParticleSwarm
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score


# Helper Functions for Data Loading
def normalize_images(x, dataset_type='mnist'):
    """
    Normalize images for a given dataset type (MNIST or CIFAR-10).
    """
    if dataset_type == 'mnist':
        return x / 255.0
    elif dataset_type == 'cifar10':
        return x / 255.0
    return x

<<<<<<< HEAD
def load_MNIST_model(model_path=None, experiment=1):
    """
    Loads a pre-trained Keras model or creates a new one if no model is provided.

    Args:
        model_path (str, optional): Path to a pre-trained Keras model. Defaults to None.
        experiment (int, optional): Defines which experiment model to create. Defaults to 1.

    Returns:
        tf.keras.Model: The Keras model.
    """
    
    if experiment == 1:
        # Create a new Keras model for experiment 1
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
    
    elif experiment == 2 or experiment == 3:
        # Create a new Keras model for experiment 2 or 3
        model = Sequential()
        
        model.add(Conv2D(32, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=3, activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=5, strides=2, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.4))

        model.add(Conv2D(128, kernel_size=4, activation='relu'))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dropout(0.4))
        model.add(Dense(10, activation='softmax'))

    if model_path is not None and os.path.isfile(model_path):
        # Load the model weights from the model path
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")

    return model

def normalize_mnist(x):
    """Applies mean/std normalization to MNIST image"""
    return x / 255.0

def load_data(batch_size=32, experiment=1):
    """
    Loads MNIST train and test data and prepares it for evaluation.

    Args:
        batch_size (int): The batch size for data loading.
        experiment (int): Experiment number to define augmentation or preprocessing.

    Returns:
        tf.data.Dataset, tf.data.Dataset: The training and testing datasets.
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize the data
    x_train = normalize_mnist(x_train.reshape(-1, 28, 28, 1).astype('float32'))
    x_test = normalize_mnist(x_test.reshape(-1, 28, 28, 1).astype('float32'))
=======
def load_data(dataset_type='mnist', batch_size=32):
    """
    Loads the specified dataset and prepares it for evaluation.
    """
    if dataset_type == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
    elif dataset_type == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize data
    x_train = normalize_images(x_train, dataset_type)
    x_test = normalize_images(x_test, dataset_type)
>>>>>>> 2e3720e3acf18d382025057cfe30b34846348776

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Apply data augmentation if experiment == 3
    if experiment == 3:
        datagen = ImageDataGenerator(
            rotation_range=10,  
            zoom_range=0.10,  
            width_shift_range=0.1, 
            height_shift_range=0.1
        )
        # Fit the datagen on the training data
        datagen.fit(x_train)

        # Create an augmented training data generator
        train_datagen = datagen.flow(x_train, y_train, batch_size=batch_size)
        
        # Wrap it into a tf.data.Dataset for compatibility with TensorFlow model training
        train_dataset = tf.data.Dataset.from_generator(
            lambda: train_datagen,
            output_signature=(
                tf.TensorSpec(shape=(batch_size, 28, 28, 1), dtype=tf.float32),
                tf.TensorSpec(shape=(batch_size, 10), dtype=tf.float32)
            )
        )
    else:
        # No augmentation, simply batch the data
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

    # Create test dataset (no augmentation applied)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    
    return train_dataset, test_dataset


# Helper Functions for Model Creation
def create_model(dataset_type='mnist'):
    """
    Create a CNN model for the specified dataset type.
    """
    if dataset_type == 'mnist':
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
    elif dataset_type == 'cifar10':
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
    return model


def load_model(model_path=None, dataset_type='mnist'):
    """
    Loads a pre-trained model or creates a new one if no model is provided.
    """
<<<<<<< HEAD
    # Convert dataset to a list of images and labels for indexing
    dataset_list = list(dataset.as_numpy_iterator())
    all_images, all_labels = zip(*dataset_list)  # Unpack images and labels
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Ensure the index is within bounds
    if image_index < 0 or image_index >= len(all_images):
        raise ValueError(f"Image index {image_index} is out of bounds. Dataset size: {len(all_images)}")

    # Select the specified image and its ground truth label
    single_image_input = all_images[image_index]
    single_image_target = np.argmax(all_labels[image_index])  # Use the actual label

    single_misclassification_target = (single_image_target + 1) % 10  # Change target to a different class

    # Ensure the targets are different to simulate misclassification
    assert single_image_target != single_misclassification_target, \
        "Target classes should be different for misclassification."

    # Create a noisy input set for black-box attack
    input_set = [single_image_input + (np.random.uniform(0, 1, single_image_input.shape) * (np.random.rand(*single_image_input.shape) < 0.9)) for _ in range(num_particles) ]
    input_set = np.stack(input_set)

    print(f"Original class: {single_image_target}")
    print(f"Misclassification target class: {single_misclassification_target}")

    # Initialize the Particle Swarm optimizer with the model and input set
    attacker = ParticleSwarm(
        model=model, input_set=input_set,starting_class=single_image_target, target_class=single_misclassification_target, num_iterations=num_iterations,
        save_dir=output_dir, inertia_weight=1, cognitive_weight=0.8,
        social_weight=0.5, momentum=0.9, clip_value_position=0.2
    )

    attacker.optimize()

    analysis(attacker, single_image_input, single_misclassification_target)
=======
    if model_path:
        return load_keras_model(model_path)
    else:
        return create_model(dataset_type)
>>>>>>> 2e3720e3acf18d382025057cfe30b34846348776

def analysis(attacker, single_misclassification_input: np.ndarray, single_misclassification_target):
    adv_img = attacker.reduce_excess_perturbations(single_misclassification_input.squeeze(), single_misclassification_target)

    for i in range(len(adv_img)):
        fig, axs = plt.subplots(1, 5, figsize=(15, 5))

        original = single_misclassification_input.squeeze().reshape(28,28)
        perturbed = np.copy(attacker.particles[i].position).reshape(28,28)

        axs[0].imshow(original, cmap="gray")
        axs[0].set_title("Original Image")

        axs[1].imshow(original - perturbed, cmap="gray")
        axs[1].set_title("Original - Perturbation")

        axs[2].imshow(perturbed, cmap="gray")
        axs[2].set_title("Perturbation")

        axs[3].imshow(perturbed - adv_img[i], cmap="gray")
        axs[3].set_title("Perturbation - Adversarial")

        axs[4].imshow(adv_img[i], cmap="gray")
        axs[4].set_title("Adversarial Image")

        for ax in axs:
            ax.axis("off")

        plt.tight_layout()
        os.makedirs(f"{attacker.save_dir}/denoised", exist_ok=True)
        plt.savefig(f"{attacker.save_dir}/denoised/{i}.png")
        plt.close(fig)

# Helper Function for Evaluation
def evaluate_model(model, test_dataset):
    """
    Evaluates the model on the test dataset and prints loss, accuracy, auROC, and auPRC.
    """
    y_true = []
    # Compute loss and accuracy
    loss, accuracy = model.evaluate(test_dataset, verbose=0)

    # Collect ground truth labels
    for images, labels in test_dataset:
        y_true.extend(np.argmax(labels.numpy(), axis=1))

    # Predict all at once to suppress excessive output
    y_pred = model.predict(test_dataset, verbose=0)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    auroc = roc_auc_score(to_categorical(y_true, num_classes=10), y_pred, multi_class='ovr')
    auprc = average_precision_score(to_categorical(y_true, num_classes=10), y_pred)

    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test auROC: {auroc:.4f}")
    print(f"Test auPRC: {auprc:.4f}")


# Helper Functions for Training
def train_model(model, train_dataset, epochs=10):
    """
    Trains the model for a specified number of epochs.
    """
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    for epoch in range(epochs):
        start_time = time.time()  # Track time for each epoch
        print(f"\nEpoch {epoch + 1}/{epochs}:")
        
        running_loss = 0.0
        running_accuracy = 0.0
        num_batches = 0

        # Use tqdm for a progress bar
        for images, labels in tqdm(train_dataset, desc="Training", unit="batch"):
            # Use train_on_batch instead of fit to update the model on each batch
            loss, accuracy = model.train_on_batch(images, labels)
            running_loss += loss
            running_accuracy += accuracy
            num_batches += 1

        epoch_loss = running_loss / num_batches
        epoch_accuracy = running_accuracy / num_batches
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {elapsed_time:.2f}s, Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    return model


# Adversarial Attack Helper
def adversarial_attack_blackbox(model, dataset, image_index, output_dir='results', num_iterations=30, num_particles=100):
    """
    Perform adversarial attack on a specific image using Particle Swarm optimization.
    """
    # Convert dataset to a list of images and labels for indexing
    dataset_list = list(dataset.as_numpy_iterator())
    all_images, all_labels = zip(*dataset_list)
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Ensure the index is within bounds
    if image_index < 0 or image_index >= len(all_images):
        raise ValueError(f"Image index {image_index} is out of bounds. Dataset size: {len(all_images)}")

    # Select the specified image and its ground truth label
    single_image_input = all_images[image_index]
    single_image_target = np.argmax(all_labels[image_index])

    single_misclassification_target = (single_image_target + 1) % 10  # Change target to a different class

    input_set = [single_image_input + (np.random.uniform(0, 1, single_image_input.shape) * (np.random.rand(*single_image_input.shape) < 0.9)) for _ in range(num_particles) ]
    input_set = np.stack(input_set)

    attacker = ParticleSwarm(model, input_set, single_misclassification_target, num_iterations=num_iterations, save_dir=output_dir, inertia_weight=.01)
    attacker.optimize()

    analysis(attacker, single_image_input, single_misclassification_target)


# Main Function to Tie Everything Together
def main():
    """
    Main function to execute the adversarial attack workflow.
    """
    # Define argument parser
    parser = argparse.ArgumentParser(description="Adversarial attack workflow with optional pre-trained Keras model.")
    parser.add_argument('--dataset', type=str, default='mnist', choices=['mnist', 'cifar10'], help="Dataset to use (mnist or cifar10).")
    parser.add_argument('--model_path', type=str, default=None, help="Path to a pre-trained Keras model.")
    parser.add_argument('--iterations', type=int, default=5, help="Number of iterations for the black-box attack.")
    parser.add_argument('--particles', type=int, default=10, help="Number of particles for the black-box attack.")
    parser.add_argument('--save_dir', type=str, default="analysis_results", help="Directory to save analysis results.")
    parser.add_argument('--model_experiment', type=int, default=1, help="Which of the 3 experiments to run: (1 is simple model, 2 is advanced, 3 is advanced with augmentation)")

    args = parser.parse_args()

<<<<<<< HEAD
    #seed_everything(1252025)

    # Load pre-trained model (MNIST model) or create a new one
    model = load_MNIST_model(args.model_path, args.model_experiment)

    # Load MNIST dataset (train and test datasets)
    train_dataset, test_dataset = load_data()

    if args.model_path is None or not os.path.isfile(args.model_path):
        # Train the model if no pre-trained model is provided
        model = train(model, train_dataset, epochs=5)
        model.save(f"mnist_model_{args.model_experiment}.keras")
        print(f"Model saved to mnist_model_{args.model_experiment}.keras")
=======
    model = load_model(args.model_path, args.dataset)
    train_dataset, test_dataset = load_data(args.dataset)

    if args.model_path is None:
        model = train_model(model, train_dataset, epochs=5)
        model.save(f'{args.dataset}_model.keras')
        print(f"Model saved to {args.dataset}_model.keras")
>>>>>>> 2e3720e3acf18d382025057cfe30b34846348776

    print("Model statistics on test dataset")
    evaluate_model(model, test_dataset)

    adversarial_attack_blackbox(model, test_dataset, image_index=0, output_dir=args.save_dir, num_iterations=args.iterations, num_particles=args.particles)


if __name__ == "__main__":
    main()
