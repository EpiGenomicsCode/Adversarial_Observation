import argparse
import os
import pickle
import tensorflow as tf
import torch
from taint import adversarial_attack_blackbox


def load_model(model_path):
    # Assumes it's a Keras model (update if using PyTorch)
    return tf.keras.models.load_model(model_path)


def get_test_dataset(data_name):
    # Import here to avoid unnecessary dependencies if unused
    from train import get_data  # Ensure get_data returns (train_ds, test_ds)

    train_ds, test_ds = get_data(data_name)
    return test_ds


def main():
    parser = argparse.ArgumentParser()

    # Required args
    parser.add_argument('--model_path', type=str, required=True, help='Path to saved model (.keras)')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save attack results')
    parser.add_argument('--source_index', type=int, required=True, help='Index of image to attack')
    parser.add_argument('--target', type=int, required=True, help='Target class for adversarial attack')

    # Dataset config
    parser.add_argument('--data', type=str, choices=['MNIST', 'MNIST_Audio'], required=True, help='Dataset name')

    # Attack config
    parser.add_argument('--iterations', type=int, default=30, help='Number of attack iterations')
    parser.add_argument('--particles', type=int, default=100, help='Number of swarm particles')

    args = parser.parse_args()

    # Load model and dataset
    model = load_model(args.model_path)
    test_ds = get_test_dataset(args.data)

    # Create output directory
    os.makedirs(args.save_dir, exist_ok=True)

    # Run the blackbox adversarial attack
    try:
        attacker = adversarial_attack_blackbox(
            model=model,
            dataset=test_ds,
            image_index=args.source_index,
            output_dir=args.save_dir,
            num_iterations=args.iterations,
            num_particles=args.particles,
            target_class=args.target
        )

        # Save attacker object
        output_path = os.path.join(args.save_dir, f'attacker_{args.source_index}_to_{args.target}.pkl')
        with open(output_path, 'wb') as f:
            pickle.dump(attacker, f)

        print(f"Attack complete. Saved attacker to: {output_path}")

    except Exception as e:
        print(f"Error during attack: {e}")


if __name__ == '__main__':
    main()
