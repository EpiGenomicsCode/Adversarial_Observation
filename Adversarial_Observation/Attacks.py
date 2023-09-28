import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def fgsm_attack(input_batch_data: torch.tensor, model: torch.nn.Module, input_shape: tuple, epsilon: float) -> torch.Tensor:
    """
    Apply the FGSM attack to input images given a pre-trained PyTorch model.

    Args:
        input_batch_data (ndarray): Batch of input images as a 4D numpy array.
        model (nn.Module): Pre-trained PyTorch model to be attacked.
        input_shape (tuple): Shape of the input array.
        epsilon (float): Magnitude of the perturbation for the attack.

    Returns:
        adversarial_batch_data (ndarray): Adversarial images generated by the FGSM attack.
    """
    # Set the model to evaluation mode
    model.eval()

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model and input batch data to the appropriate device
    model.to(device)
    input_batch_data = torch.tensor(input_batch_data).to(device)

    # Disable gradient computation for all model parameters
    for param in model.parameters():
        param.requires_grad = False

    adversarial_batch_data = []
    for img in input_batch_data:
        # Convert the input image to a PyTorch tensor with dtype=torch.float32 and enable gradient computation
        img = img.clone().detach().to(torch.float32).requires_grad_(True)

        # Move the input image tensor to the same device as the model
        img = img.to(device)

        # Make a forward pass through the model and get the predicted class scores for the input image
        preds = model(img.reshape(input_shape))

        # Compute the loss by selecting the class with the highest predicted score
        target = torch.argmax(preds)
        loss = torch.nn.functional.cross_entropy(preds, target.unsqueeze(0))

        # Compute gradients of the loss with respect to the input image pixels
        model.zero_grad()
        loss.backward()

        # Calculate the sign of the gradients
        gradient_sign = img.grad.sign()

        # Create the adversarial example by adding the signed gradients to the original image
        adversarial_img = img + epsilon * gradient_sign

        # Clip the adversarial image to ensure pixel values are within the valid range
        adversarial_img = torch.clamp(adversarial_img, 0, 1)

        adversarial_batch_data.append(adversarial_img.cpu().detach().numpy())

    return adversarial_batch_data


def gradient_map(input_batch_data: torch.tensor, model: torch.nn.Module, input_shape: tuple, backprop_type: str = 'guided') -> torch.Tensor:
    """
    Generate a gradient map for an input image given a pre-trained PyTorch model.

    Args:
        input_batch_data (ndarray): Batch of input images as a 4D numpy array.
        model (nn.Module): Pre-trained PyTorch model used to generate the gradient map.
        input_shape (tuple): Shape of the input array.
        backprop_type (str, optional): Type of backpropagation. Supported values: 'vanilla', 'guided', 'relu'.
            Defaults to 'vanilla'.

    Returns:
        gradient_maps (ndarray): Gradient map for the input images.
    """
    # Set the model to evaluation mode
    model.eval()

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model and input batch data to the appropriate device
    model.to(device)
    input_batch_data = torch.tensor(input_batch_data).to(device)

    # Disable gradient computation for all model parameters
    for param in model.parameters():
        param.requires_grad = False

    gradient_maps = []
    for img in input_batch_data:
        # Convert the input image to a PyTorch tensor with dtype=torch.float32 and enable gradient computation
        img = img.clone().detach().to(torch.float32).requires_grad_(True)

        # Move the input image tensor to the same device as the model
        img = img.to(device)

        # Make a forward pass through the model and get the predicted class scores for the input image
        preds = model(img.reshape(input_shape))

        # Compute the score and index of the class with the highest predicted score
        score, _ = torch.max(preds, 1)

        # Reset gradients from previous iterations
        model.zero_grad()

        # Compute gradients of the score with respect to the model parameters
        score.backward()

        if backprop_type == 'guided':
            # Apply guided backpropagation
            gradients = img.grad
            gradients = gradients * (gradients > 0).float()  # ReLU-like operation
            gradient_map = gradients.norm(dim=0)
        elif backprop_type == 'relu':
            # Apply ReLU backpropagation
            gradients = img.grad
            gradients = (gradients > 0).float()  # ReLU operation
            gradient_map = gradients.norm(dim=0)
        else:
            # Default to vanilla backpropagation
            gradient_map = img.grad.norm(dim=0)

        gradient_maps.append(gradient_map.cpu().detach().numpy())

    return gradient_maps


def gradient_ascent(input_batch_data: torch.tensor, model: torch.nn.Module, input_shape: tuple, target_neuron: int, num_iterations: int = 100, step_size: float = 1.0) -> torch.Tensor:
    """
    Perform gradient ascent to generate an image that maximizes the activation of a target neuron given a pre-trained PyTorch model.

    Args:
        input_batch_data (ndarray): Batch of input images as a 4D numpy array.
        model (nn.Module): Pre-trained PyTorch model used for gradient ascent.
        input_shape (tuple): Shape of the input array.
        target_neuron (int): Index of the target neuron to maximize activation.
        num_iterations (int, optional): Number of gradient ascent iterations. Defaults to 100.
        step_size (float, optional): Step size for each iteration. Defaults to 1.0.

    Returns:
        generated_images (ndarray): Generated images that maximize the activation of the target neuron.
    """
    # Set the model to evaluation mode
    model.eval()

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model to the appropriate device
    model.to(device)

    # Initialize the generated images
    generated_images = []
    for img in input_batch_data:
        # Convert the input image to a PyTorch tensor with dtype=torch.float32 and enable gradient computation
        img = img.clone().detach().to(torch.float32).requires_grad_(True)

        # Move the input image tensor to the same device as the model
        img = img.to(device)

        # Perform gradient ascent
        for _ in range(num_iterations):
            # Make a forward pass through the model and get the activation of the target neuron
            activation = model(img.reshape(input_shape))[:, target_neuron]

            # Reset gradients from previous iterations
            model.zero_grad()

            # Compute gradients of the target neuron activation with respect to the input image using torch.autograd.grad
            gradients = torch.autograd.grad(activation, img)[0]

            # Normalize the gradients
            gradients = F.normalize(gradients, dim=0)

            # Update the input image using the gradients and step size
            img.data += step_size * gradients

        # Append the generated image to the list
        generated_images.append(img.cpu().detach().numpy())

    return generated_images

def saliency_map(input_image, model, target_class=None):
    """
    Generate a saliency map for an input image given a pre-trained PyTorch model.

    Args:
        input_image (torch.Tensor): Input image as a 3D torch.Tensor.
        model (torch.nn.Module): Pre-trained PyTorch model used to generate the saliency map.
        target_class (int, optional): Index of the target class for saliency computation.
            If None, the class with the highest predicted score will be used.

    Returns:
        saliency_map (torch.Tensor): Saliency map for the input image.
    """
    # Set the model to evaluation mode
    model.eval()

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move the model and input image to the appropriate device
    model.to(device)
    input_image = input_image.to(device).requires_grad_()

    # Make a forward pass through the model and get the predicted class scores for the input image
    logits = model(input_image.unsqueeze(0))  # Add a batch dimension
    if target_class is None:
        # If target class is not specified, use the class with the highest predicted score
        _, target_class = torch.max(logits, 1)

    # Compute the score for the target class
    target_score = logits[0, target_class]

    # Compute gradients of the target class score with respect to the input image
    target_score.backward()

    # Calculate the absolute gradients as the saliency map
    saliency_map = input_image.grad.abs().squeeze(0).cpu()

    return saliency_map