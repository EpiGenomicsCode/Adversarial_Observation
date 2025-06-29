import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import models

# Ensures reproducibility by setting the random seed across all libraries
def seed_everything(seed: int):
    """
    Sets the seed for random number generation to ensure reproducibility.

    Args:
        seed (int): The seed value to use.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# FGSM Attack (Fast Gradient Sign Method)
def fgsm_attack(input_data: torch.Tensor, model: torch.nn.Module, epsilon: float, device: torch.device) -> torch.Tensor:
    """
    Performs FGSM attack on the input data.
    
    Args:
        input_data (torch.Tensor): The original input batch.
        model (torch.nn.Module): The model to attack.
        epsilon (float): The perturbation magnitude.
        device (torch.device): The device to perform the attack on (cuda or cpu).
        
    Returns:
        torch.Tensor: The adversarially perturbed batch.
    """
    input_data.requires_grad = True

    # Forward pass to get model's output
    output = model(input_data)

    # Compute the loss (we assume classification, so we use cross-entropy loss)
    loss = F.cross_entropy(output, torch.argmax(output, dim=1))

    # Backpropagate the gradients
    model.zero_grad()
    loss.backward()

    # Get the gradient of the input data with respect to the loss
    gradient = input_data.grad.data

    # Create the adversarial example by perturbing the input data
    adversarial_data = input_data + epsilon * torch.sign(gradient)
    
    # Ensure the adversarial data is within the valid range (e.g., [0, 1] for image data)
    adversarial_data = torch.clamp(adversarial_data, 0, 1)
    
    return adversarial_data

# Compute the success rate of the attack
def compute_success_rate(original_preds: torch.Tensor, adversarial_preds: torch.Tensor) -> float:
    """
    Computes the success rate of the attack, which is the fraction of adversarial examples where the model
    was misled (i.e., adversarial prediction does not match the original prediction).
    
    Args:
        original_preds (torch.Tensor): The original model predictions.
        adversarial_preds (torch.Tensor): The predictions on adversarial examples.
        
    Returns:
        float: The success rate of the attack.
    """
    success = (original_preds != adversarial_preds).sum().item()
    return success / len(original_preds)

# Visualize and save adversarial examples
def visualize_adversarial_examples(original_images: torch.Tensor, adversarial_images: torch.Tensor, original_image_path: str, adversarial_image_path: str):
    """
    Visualizes and saves the original and adversarial images side by side.

    Args:
        original_images (torch.Tensor): Original input images.
        adversarial_images (torch.Tensor): Adversarially perturbed images.
        original_image_path (str): Path to save the original images.
        adversarial_image_path (str): Path to save the adversarial images.
    """
    # Convert tensors to numpy arrays (for visualization)
    original_images = original_images.cpu().detach().numpy()
    adversarial_images = adversarial_images.cpu().detach().numpy()

    # Set up the plotting grid
    num_images = original_images.shape[0]
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))

    for i in range(num_images):
        # Plot the original image
        axes[i].imshow(np.transpose(original_images[i], (1, 2, 0)))
        axes[i].set_title("Original")
        axes[i].axis('off')

    # Save the figure with original images
    plt.tight_layout()
    plt.savefig(original_image_path)
    plt.close(fig)

    # Set up the plotting grid for adversarial images
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))

    for i in range(num_images):
        # Plot the adversarial image
        axes[i].imshow(np.transpose(adversarial_images[i], (1, 2, 0)))
        axes[i].set_title("Adversarial")
        axes[i].axis('off')

    # Save the figure with adversarial images
    plt.tight_layout()
    plt.savefig(adversarial_image_path)
    plt.close(fig)

# Log metrics such as success rate and average perturbation
def log_metrics(success_rate: float, avg_perturbation: float):
    """
    Logs the success rate and average perturbation of the attack.
    
    Args:
        success_rate (float): The success rate of the attack.
        avg_perturbation (float): The average perturbation magnitude.
    """
    logging.info(f"Attack Success Rate: {success_rate:.4f}")
    logging.info(f"Average Perturbation: {avg_perturbation:.4f}")

def load_MNIST_model():
    """
    Loads a sequential CNN model for MNIST dataset.

    Returns:
        torch.nn.Module: The CNN model.
    """
    model = torch.nn.Sequential(
        torch.nn.Conv2d(1, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2),
        torch.nn.Flatten(),
        torch.nn.Linear(64 * 7 * 7, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 10)
    )

    return model

def load_data(batch_size=32):
    """
    Loads MNIST train and test data and prepares it for evaluation.

    Args:
        batch_size (int): The batch size for data loading.

    Returns:
        TrinLoader, TestLoader: The training and testing data loaders.
    """
    # Define the transformations for the dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Create data loaders for the training and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader