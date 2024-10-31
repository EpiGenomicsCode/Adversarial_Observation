import torch
import torch.nn as nn
import numpy as np
import torchvision

def load_MNIST_model() -> nn.Sequential:
    """
    Build a convolutional neural network model.

    Returns:
        nn.Sequential: A convolutional neural network model for MNIST classification.
    """
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Flatten(),
        nn.Linear(7 * 7 * 16, 10),
        nn.Softmax(dim=1)
    )

def seed_everything(seed: int) -> None:
    """
    Seed the random number generators for reproducibility.

    Args:
        seed (int): The seed for random number generation.

    Returns:
        None
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_MNIST_data():
    """
    Load the MNIST dataset and create data loaders.

    Returns:
        tuple: (train_loader, test_loader) - Data loaders for training and testing.
    """
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    return train_loader, test_loader
