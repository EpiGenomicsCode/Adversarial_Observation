import torch
import torch.nn as nn
import numpy as np
import torchvision

def load_MNIST_model() -> nn.Sequential:
    """
    Builds a convolutional neural network model.

    Args:
        output_size (int): The number of output classes.
    
    Returns:
        model (torch.nn.Sequential): A convolutional neural network model with the specified output size.
    """
    # Create a sequential container for building the model
    return nn.Sequential(
        # Add a 2D convolutional layer with 1 input channel, 8 output channels, 3x3 kernel size,
        # 1 pixel padding, and 1 pixel stride.
        nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
        # Add a ReLU activation layer.
        nn.ReLU(),
        # Add a 2D max pooling layer with 2x2 kernel size and 2 pixel stride.
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Add another 2D convolutional layer with 8 input channels, 16 output channels, 3x3 kernel size,
        # 1 pixel padding, and 1 pixel stride.
        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        # Add another ReLU activation layer.
        nn.ReLU(),
        # Add another 2D max pooling layer with 2x2 kernel size and 2 pixel stride.
        nn.MaxPool2d(kernel_size=2, stride=2),
        # Add a flattening layer to convert the 2D feature maps to a 1D vector.
        nn.Flatten(),
        # Add a fully connected layer with 7x7x16 input features and output_size output features.
        nn.Linear(7*7*16, 10),
        # Add a softmax activation layer to convert the outputs to probabilities.
        nn.Softmax(dim=1)
    )

def seedEverything(seed: int) -> None:
    """
    Seeds the random number generators for Python, PyTorch, and CUDA to make the results reproducible.

    Args:
        seed (int): The seed to use for random number generation.

    Returns:
        None
    """
    # Seed NumPy for random number generation.
    np.random.seed(seed)
    # Seed PyTorch for random number generation.
    torch.manual_seed(seed)
    # Seed CUDA for random number generation (if available).
    torch.cuda.manual_seed_all(seed)

    return None

def load_MNIST_data():
    # read in the MNIST data from torchvision.datasets
    train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=torchvision.transforms.ToTensor())
    test_data = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=torchvision.transforms.ToTensor())

    # create data loaders for the training and test data
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)

    return train_loader, test_loader