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

# PGD Attack (Projected Gradient Descent)
def pgd_attack(input_data: torch.Tensor, model: torch.nn.Module, epsilon: float, alpha: float, num_steps: int, device: torch.device) -> torch.Tensor:
    """
    Performs PGD attack on the input data.
    
    Args:
        input_data (torch.Tensor): The original input batch.
        model (torch.nn.Module): The model to attack.
        epsilon (float): The maximum perturbation magnitude.
        alpha (float): The step size for each iteration.
        num_steps (int): The number of steps for the attack.
        device (torch.device): The device to perform the attack on (cuda or cpu).
        
    Returns:
        torch.Tensor: The adversarially perturbed batch.
    """
    # Initialize the perturbation to be the same as the original input data
    perturbed_data = input_data.clone().detach()
    perturbed_data.requires_grad = True

    for _ in range(num_steps):
        # Forward pass to get the model's output
        output = model(perturbed_data)
        
        # Compute the loss (we assume classification, so we use cross-entropy loss)
        loss = F.cross_entropy(output, torch.argmax(output, dim=1))

        # Backpropagate the gradients
        model.zero_grad()
        loss.backward()

        # Get the gradient of the input data with respect to the loss
        gradient = perturbed_data.grad.data

        # Update the perturbation (step in the direction of the gradient)
        perturbed_data = perturbed_data + alpha * torch.sign(gradient)
        
        # Clip the perturbation to stay within the epsilon ball
        perturbed_data = torch.max(torch.min(perturbed_data, input_data + epsilon), input_data - epsilon)

        # Ensure the adversarial data is within the valid range (e.g., [0, 1] for image data)
        perturbed_data = torch.clamp(perturbed_data, 0, 1)

    return perturbed_data

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

def load_pretrained_model():
    """
    Loads a pre-trained model (e.g., ResNet18) for evaluation.

    Returns:
        torch.nn.Module: The pre-trained model (ResNet18).
    """
    model = models.resnet18(weights='IMAGENET1K_V1')  # Ensure correct weights argument is used
    model.eval()  # Set the model to evaluation mode
    return model

def load_data(batch_size=32):
    """
    Loads CIFAR-10 validation data and prepares it for evaluation.

    Args:
        batch_size (int): The batch size for data loading.

    Returns:
        DataLoader: A DataLoader object for the CIFAR-10 validation dataset.
    """
    # Define the transformation for image preprocessing (same as what was used to train the model)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet mean and std
    ])

    # Use CIFAR-10 dataset instead of ImageNet for simplicity
    dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Use a DataLoader for batching
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader
