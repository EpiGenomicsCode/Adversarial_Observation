import argparse
import os
from train import load_data, train_model, evaluate_model, train_model_and_save

def main():
    parser = argparse.ArgumentParser(description="Train and evaluate MNIST or AudioMNIST models")

    # Dataset and model type
    parser.add_argument('--data', type=str, choices=['MNIST', 'MNIST_Audio'], required=True,
                        help='Dataset to use: MNIST or MNIST_Audio')
    parser.add_argument('--model_type', type=str,
                        choices=['normal', 'complex', 'complex_augmented', 'complex_adversarial'],
                        required=True, help='Model architecture type')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')

    # Adversarial training option
    parser.add_argument('--adversarial', type=str, default='none',
                        choices=['none', 'pgd', 'trades'],
                        help='Adversarial training method to use (if any)')

    # Output directory
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save trained models and results')

    args = parser.parse_args()

    # Run training + evaluation
    train_model_and_save(args)

if __name__ == '__main__':
    main()
