import argparse
import os
import numpy as np
import pickle
import tensorflow as tf
import json
from tqdm import tqdm
from taint import adversarial_attack_blackbox
from train import train_model_and_save
from analysis import get_softmax_stats, save_softmax_stats


def collect_statistics(model, dataset, model_type, attack_iterations=10, attack_particles=100, image_index=0, output_dir='results'):
    """
    Run adversarial attack for the given model and dataset combination and collect statistics.

    Args:
    - model: The model to attack.
    - dataset: The test dataset.
    - model_type: The model type (for logging).
    - attack_iterations: Number of iterations for attack.
    - attack_particles: Number of particles for attack.
    - image_index: Index of the image to perform the attack on.
    - output_dir: Directory to save results.

    Returns:
    - statistics: A dictionary with softmax output, attack success, and other relevant data.
    """
    pickle_path = os.path.join(output_dir, f"{model_type}_attacker.pkl")

    dataset_list = list(dataset.as_numpy_iterator())
    all_images, all_labels = zip(*dataset_list)
    all_images = np.concatenate(all_images, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    if image_index < 0 or image_index >= len(all_images):
        raise ValueError(f"Image index {image_index} out of range")

    single_input = all_images[image_index]
    single_target = np.argmax(all_labels[image_index])
    target_class = (single_target + 1) % 10  # Attack a different class

    # Load the attacker if pickle exists
    if os.path.exists(pickle_path):
        with open(pickle_path, 'rb') as f:
            attacker = pickle.load(f)
        print(f"Loaded attacker from {pickle_path}")
    else:
        # If attacker doesn't exist, run the attack and save it
        adversarial_attack_blackbox(
            model, dataset, image_index=image_index, output_dir=output_dir,
            num_iterations=attack_iterations, num_particles=attack_particles
        )
        with open(pickle_path, 'wb') as f:
            pickle.dump(attacker, f)
        print(f"Saved attacker to {pickle_path}")

    # Analyze the attack results
    softmax_output, max_val, max_class = get_softmax_stats(model, single_input)
    attack_success = max_class != target_class  # Attack success is when max_class differs from target class

    stats = {
        "model_type": model_type,
        "target_class": target_class,
        "attack_success": attack_success,
        "softmax_output": softmax_output.tolist(),
        "max_confidence": max_val,
        "max_class": max_class,
    }

    # Save softmax statistics for this model and image
    save_softmax_stats(os.path.join(output_dir, f"{model_type}_softmax_stats.tsv"), softmax_output, max_class, max_val, target_class)

    return stats


def get_model_types_for_dataset(dataset):
    """
    Dynamically search for model types for a given dataset.

    Args:
    - dataset: The dataset name, e.g., 'MNIST' or 'AudioMNIST'.

    Returns:
    - model_types: List of available model types for the dataset.
    """
    model_types = ['normal', 'complex', 'complex_augmented']  # Can be extended if needed

    return model_types


def run_statistics(args):
    # Define output directory to save results
    results_dir = os.path.join(args.save_dir, f"{args.data}_stats")
    os.makedirs(results_dir, exist_ok=True)

    # Store statistics for all combinations of dataset and model types
    all_stats = []

    # Get all possible model types for the dataset
    model_types = get_model_types_for_dataset(args.data)

    # Iterate through all pairs of model types
    for model_type in model_types:
        print(f"Training model: {model_type}...")
        model, test_ds, _, model_path = train_model_and_save(args)  # Train the model for the current type

        # Run adversarial attack and collect stats for each combination of different model types
        for other_model_type in model_types:
            if model_type != other_model_type:  # Skip 1-1 combinations
                print(f"Attacking {other_model_type} model with {model_type} dataset...")
                stats = collect_statistics(model, test_ds, other_model_type, attack_iterations=args.iterations, attack_particles=args.particles, output_dir=results_dir)
                all_stats.append(stats)

    # Save the collected statistics as a JSON file for later analysis
    stats_file = os.path.join(results_dir, f"{args.data}_attack_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=4)   

    print(f"Statistics saved to {stats_file}")


def main():
    # Command-line arguments
    parser = argparse.ArgumentParser()

    # Data and model type arguments
    parser.add_argument('--data', type=str, choices=['MNIST', 'MNIST_Audio'], required=True, help='Dataset to use')
    parser.add_argument('--model_type', type=str, choices=['normal', 'complex', 'complex_augmented'], required=True, help='Model type to use')

    # Attack parameters
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for attack')
    parser.add_argument('--particles', type=int, default=100, help='Number of particles for attack')

    # Folder saving argument
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save model and results')

    # Parse arguments
    args = parser.parse_args()

    # Run the statistics collection
    run_statistics(args)


if __name__ == '__main__':
    main()
