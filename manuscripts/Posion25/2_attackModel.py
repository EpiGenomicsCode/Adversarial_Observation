import argparse
import os
import pickle
from taint import adversarial_attack_blackbox
from analysis import *
from train import train_model_and_save
import torch
import tensorflow as tf

def attack_model(args, model, test_ds, save_dir, num_data=10):
    # Get the labels by iterating through a batch from the test_ds
    first_batch = next(iter(test_ds))  # Get the first batch
    images, labels = first_batch  # Unpack the images and labels from the first batch
    
    # Check if labels are a TensorFlow tensor or PyTorch tensor
    if isinstance(labels, tf.Tensor):
        # If using TensorFlow, convert labels to class indices (from one-hot encoded)
        labels = tf.argmax(labels, axis=1).numpy()  # Get class indices from one-hot encoded labels
    elif isinstance(labels, torch.Tensor):
        # If using PyTorch, convert labels to class indices (from one-hot encoded)
        labels = torch.argmax(labels, dim=1).cpu().numpy()  # Get class indices from one-hot encoded labels

    # Convert labels to a set of unique outputs
    unique_outputs = set(labels)  # Convert to a Python set for unique labels

    # Continue with the rest of the attack logic
    for output in unique_outputs:
        instances = [i for i, label in enumerate(labels) if label == output][:num_data]  # Select `num_data` instances with the current output label
        
        for image_index in instances:
            # Create a subdirectory for each image_index and its original output label
            sub_dir = os.path.join(save_dir, f'image_{image_index}_label_{output}')
            
            # Ensure the directory exists
            os.makedirs(sub_dir, exist_ok=True)

            # Correct dynamic pickle filename to include the original and target class
            pickle_filename = f'attacker_{image_index}_{output}.pkl'
            pickle_path = os.path.join(sub_dir, pickle_filename)
            
            # Check if the attacker pickle already exists for this image_index and output
            if os.path.exists(pickle_path):
                with open(pickle_path, 'rb') as f:
                    attacker = pickle.load(f)
                print(f"Loaded attacker for image {image_index} with label {output} from {pickle_path}")
            else:
                print(f"Running adversarial attack for image {image_index} with label {output}...")
                
                # For the current `output`, target all other classes
                for target_output in unique_outputs:
                    if target_output != output:  # We want to target all other outputs
                        for _ in range(num_data):  # Attack the target output `num_data` times
                            target_sub_dir = os.path.join(sub_dir, f'target_{target_output}')
                            os.makedirs(target_sub_dir, exist_ok=True)  # Create a subdir for each target class

                            # Correct dynamic pickle filename to include the original and target class
                            target_pickle_filename = f'attacker_{image_index}_{output}_to_{target_output}.pkl'
                            target_pickle_path = os.path.join(target_sub_dir, target_pickle_filename)

                            # Perform the adversarial attack targeting `target_output`
                            attacker = adversarial_attack_blackbox(
                                model=model,
                                dataset=test_ds,
                                image_index=image_index,
                                output_dir=target_sub_dir,
                                num_iterations=args.iterations,
                                num_particles=args.particles,
                                target_class=target_output  # Specify the target class for the attack
                            )
                            print(f"Adversarial attack completed for image {image_index} targeting class {target_output}")

                            # After performing the attack, save the attacker object to a pickle file
                            with open(target_pickle_path, 'wb') as f:
                                pickle.dump(attacker, f)
                            print(f"Saved attacker for image {image_index} with label {output} targeting {target_output} to {target_pickle_path}")

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
