import argparse
import os
from train import load_data, train_model, evaluate_model, train_model_and_save


def main():
    # Command-line arguments
    parser = argparse.ArgumentParser()

    # Data and model type arguments
    parser.add_argument('--data', type=str, choices=['MNIST', 'MNIST_Audio'], required=True, help='Dataset to use')
    parser.add_argument('--model_type', type=str, choices=['normal', 'complex', 'complex_augmented'], required=True, help='Model type to use')

    # Training information arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')

    # Folder saving argument
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save model and results')

    # Parse arguments
    args = parser.parse_args()

    # Train the model and evaluate it
    train_model_and_save(args)

if __name__ == '__main__':
    main()
