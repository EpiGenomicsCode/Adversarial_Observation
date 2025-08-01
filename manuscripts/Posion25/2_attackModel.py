import argparse
import os
import pickle
from taint import adversarial_attack_blackbox
from analysis import *
from train import train_model_and_save

def attack_model(args, model, test_ds, save_dir, num_data=10):
    # Path to the pickle file that stores the attacker object
    pickle_path = os.path.join(save_dir, 'attacker.pkl')
    
    # Check if the adversarial attack has already been performed (if pickle exists)
    if os.path.exists(pickle_path):
        # If pickle exists, load the attacker from the file
        with open(pickle_path, 'rb') as f:
            attacker = pickle.load(f)
        print(f"Loaded attacker from {pickle_path}")
    else:
        # If pickle does not exist, run the attack and save the attacker
        print("Running adversarial attack...")

        # First, identify unique outputs in the dataset
        unique_outputs = set(test_ds.labels)  # assuming `test_ds.labels` contains the true labels

        for output in unique_outputs:
            # Find the first 10 instances of this output in the dataset
            instances = [i for i, label in enumerate(test_ds.labels) if label == output][:num_data]            
            # Perform the attack on each of these instances
            for image_index in instances:
                adversarial_attack_blackbox(
                    model, test_ds, image_index=image_index, output_dir=save_dir,
                    num_iterations=args.iterations, num_particles=args.particles
                )
                print(f"Attacked image {image_index} with label {output}")

def main():
    # Command-line arguments
    parser = argparse.ArgumentParser()

    # Data and model type arguments (to align with the ones used in the training script)
    parser.add_argument('--data', type=str, choices=['MNIST', 'MNIST_Audio'], required=True, help='Dataset to use')
    parser.add_argument('--model_type', type=str, choices=['normal', 'complex', 'complex_augmented'], required=True, help='Model type to use')

    # Attack parameters
    parser.add_argument('--iterations', type=int, default=10, help='Number of iterations for attack')
    parser.add_argument('--particles', type=int, default=100, help='Number of particles for attack')

    # Folder saving argument
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save model and results')

    # Parse arguments
    args = parser.parse_args()

    # First, train the model and get the necessary details for attack
    model, test_ds, save_dir, model_path = train_model_and_save(args)

    # Perform the adversarial attack
    attack_model(args, model, test_ds, save_dir)

if __name__ == '__main__':
    main()
