import argparse
import time
import numpy as np
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.models import load_model as load_keras_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from Adversarial_Observation.utils import seed_everything
from Adversarial_Observation import AdversarialTester, ParticleSwarm

from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt

import sys, os, json

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

def load_MNIST_model(model_path=None):
    """
    Loads a pre-trained Keras model or creates a new one if no model is provided.

    Args:
        model_path (str, optional): Path to a pre-trained Keras model. Defaults to None.

    Returns:
        tf.keras.Model: The Keras model.
    """
    if model_path:
        # Load a pre-trained Keras model
        model = load_keras_model(model_path)
        return model
    else:
        # Create a new Keras model
        model = Sequential([
            Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(10, activation='softmax')
        ])
        return model

def normalize_mnist(x):
    """Applies mean/std normalization to MNIST image"""
#    mean = 0.1307
#    std = 0.3081
#    return ((x / 255.0) - mean) / std
    return x / 255.0

def load_data(batch_size=32):
    """
    Loads MNIST train and test data and prepares it for evaluation.

    Args:
        batch_size (int): The batch size for data loading.

    Returns:
        tf.data.Dataset, tf.data.Dataset: The training and testing datasets.
    """
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape and normalize the data
    x_train = normalize_mnist(x_train.reshape(-1, 28, 28, 1).astype('float32'))
    x_test = normalize_mnist(x_test.reshape(-1, 28, 28, 1).astype('float32'))

    # One-hot encode the labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_dataset, test_dataset

def train(model: tf.keras.Model, train_dataset: tf.data.Dataset, epochs: int = 10) -> tf.keras.Model:
    """
    Trains the model for a specified number of epochs.

    Args:
        model (tf.keras.Model): The model to train.
        train_dataset (tf.data.Dataset): The training dataset.
        epochs (int, optional): Number of training epochs. Defaults to 10.

    Returns:
        tf.keras.Model: The trained model.
    """
    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    for epoch in range(epochs):
        start_time = time.time()  # Track time for each epoch
        print(f"\nEpoch {epoch + 1}/{epochs}:")

        running_loss = 0.0
        running_accuracy = 0.0
        num_batches = 0

        # Use tqdm for a progress bar
        for images, labels in tqdm(train_dataset, desc="Training", unit="batch"):
            loss, accuracy = model.fit(images, labels)
            running_loss += loss
            running_accuracy += accuracy
            num_batches += 1

        # Print average loss and accuracy for the epoch
        epoch_loss = running_loss / num_batches
        epoch_accuracy = running_accuracy / num_batches
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch + 1} completed in {elapsed_time:.2f}s, Average Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

    return model

def adversarial_attack_blackbox(model: tf.keras.Model, dataset: tf.data.Dataset, image_index: int, output_dir: str = 'results', num_iterations: int = 30, num_particles: int = 100) -> tf.data.Dataset:
    """
    Performs a black-box adversarial attack on a specific image in the dataset using Particle Swarm optimization.

    Args:
        model (tf.keras.Model): The trained model to attack.
        dataset (tf.data.Dataset): The dataset containing the images.
        image_index (int): The index of the image in the dataset to attack.
        num_iterations (int, optional): Number of iterations for the attack. Defaults to 30.
        num_particles (int, optional): Number of particles for the attack. Defaults to 100.

    Returns:
        tf.data.Dataset: A dataset containing adversarially perturbed images.
    """
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
        model, input_set, single_misclassification_target, num_iterations=num_iterations,
        save_dir=output_dir, inertia_weight=1, cognitive_weight=0.8,
        social_weight=0.5, momentum=0.9, clip_value_position=0.2
    )

    attacker.optimize()

    analysis(attacker, single_image_input, single_misclassification_target)

def analysis(attacker, single_misclassification_input: np.ndarray, single_misclassification_target):
    """
    Analyzes the results of the attack and generates plots.
    - Saves the original misclassification input and target.
    - For each particle and each position in the particle's history:
        - Save the position (perturbed image).
        - Save all confidence values.
        - Save the maximum output (softmax confidence).
        - Save the difference from the original input.
    """
    # Save the original image and its classification
    plt.imsave(os.path.join(attacker.save_dir, "original.png"), single_misclassification_input.squeeze(), cmap="gray", vmin=0, vmax=1)
    
    analysis_results = {
        "original_misclassification_input": single_misclassification_input.tolist(),
        "original_misclassification_target": int(single_misclassification_target),
        "particles": []
    }
    
    # Process each particle in the attacker's particles list
    for particle_idx, particle in enumerate(attacker.particles):
        print(f"Processing particle: {particle_idx}")
        particle_data = {
            "particle_index": particle_idx,
            "positions": [],
            "confidence_values": [],
            "max_output_values": [],
            "max_output_classes": [],
            "differences_from_original": [],
            "confidence_over_time": []  # Store confidence over time
        }
        
        for step_idx, position in enumerate(particle.history):
            # Ensure 'position' is a numpy array.
            if isinstance(position, tf.Tensor):
                position_np = position.numpy()
            else:
                position_np = np.array(position)
            
            output = attacker.model(position_np)

            # Remove the batch dimension and apply softmax
            softmax_output = tf.nn.softmax(tf.squeeze(output), axis=0)
            confidence_values = softmax_output.numpy().tolist()
            max_output_value = float(max(confidence_values))
            max_output_class = confidence_values.index(max_output_value)

            # Calculate pixel-wise difference from original image (before attack)
            #diff_image = np.abs(position_np - single_misclassification_input)[0]
            diff_image = (position_np - single_misclassification_input)[0]
            #print(position_np)
            #print(single_misclassification_input)
            #print(diff_image)
            # Save the difference image
            iteration_folder = os.path.join(attacker.save_dir, f"iteration_{step_idx + 1}")
            if not os.path.exists(iteration_folder):
                os.makedirs(iteration_folder)
            plt.imsave(os.path.join(iteration_folder, f"attack-vector_image_{particle_idx + 1}.png"), diff_image.squeeze(), cmap="seismic", vmin=-1, vmax=1)

            # Calculate difference from original image (before attack)
            difference_from_original = float(np.linalg.norm(position - single_misclassification_input))

            # Add data for this step to the particle_data
            particle_data["positions"].append(position_np.tolist())
            particle_data["confidence_values"].append(confidence_values)
            particle_data["max_output_values"].append(max_output_value)
            particle_data["max_output_classes"].append(max_output_class)
            particle_data["differences_from_original"].append(difference_from_original)
            particle_data["confidence_over_time"].append(max_output_value)  # Store max output (confidence)
        
        # Append the particle's data to the main analysis results
        analysis_results["particles"].append(particle_data)
    
    # Save the analysis results to a JSON file
    output_dir = attacker.save_dir  # Use the save_dir from the attacker
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "attack_analysis.json")
    
    with open(file_path, "w") as f:
        json.dump(analysis_results, f, indent=4)
    
    print(f"Analysis results saved to {file_path}")

def main() -> None:
    """
    Main function to execute the adversarial attack workflow.
    """
    parser = argparse.ArgumentParser(description="Adversarial attack workflow with optional pre-trained Keras model.")
    parser.add_argument('--model_path', type=str, default=None, help="Path to a pre-trained Keras model.")
    parser.add_argument('--iterations', type=int, default=50, help="Number of iterations for the black-box attack.")
    parser.add_argument('--particles', type=int, default=100, help="Number of particles for the black-box attack.")
    parser.add_argument('--save_dir', type=str, default="analysis_results", help="Directory to save analysis results.")
    args = parser.parse_args()

    #seed_everything(1252025)

    # Load pre-trained model (MNIST model) or create a new one
    model = load_MNIST_model(args.model_path)

    # Load MNIST dataset (train and test datasets)
    train_dataset, test_dataset = load_data()

    if args.model_path is None:
        # Train the model if no pre-trained model is provided
        model = train(model, train_dataset, epochs=5)
        model.save('mnist_model.keras')
        print("Model saved to mnist_model.keras")

    # Evaluate the model
    print("Model statistics on test dataset")
    evaluate_model(model, test_dataset)

    # Perform adversarial attack
    adversarial_dataset = adversarial_attack_blackbox(model, test_dataset, 0, output_dir=args.save_dir, num_iterations=args.iterations, num_particles=args.particles)

if __name__ == "__main__":
    main()