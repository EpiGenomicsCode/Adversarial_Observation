import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def activation_map(input_data: torch.Tensor, model: torch.nn.Module, normalize: bool = True) -> np.ndarray:
    """
    Generate an activation map for input data given a pre-trained PyTorch model.

    Args:
        input_data (torch.Tensor): Input data as a PyTorch tensor.
            Shape: (batch_size, channels, height, width)
        model (torch.nn.Module): Pre-trained PyTorch model used to generate the activation map.
        normalize (bool, optional): Flag to normalize the activation map. Default is True.

    Returns:
        activation_map (np.ndarray): Activation map for the input data based on the input model.
            Shape: (batch_size, height, width)
    """
    model.eval()  # Set the model to evaluation mode
    assert isinstance(input_data, torch.Tensor), "Input data must be a PyTorch tensor"
    assert input_data.device == next(model.parameters()).device, "Input data and model must be on the same device"

    with torch.enable_grad():
        input_data.requires_grad_(True)  # Enable gradient computation for the input data

        output = model(input_data)  # Forward pass the input data through the model

        gradients = torch.autograd.grad(output.sum(), input_data)[0]  # Compute the gradients

    activation_map = gradients.detach().cpu().numpy()  # Convert the activation map to a numpy array

    if normalize:
        min_vals = np.min(activation_map, axis=(1, 2), keepdims=True)
        max_vals = np.max(activation_map, axis=(1, 2), keepdims=True)
        activation_map = (activation_map - min_vals) / (max_vals - min_vals + 1e-8)


    return activation_map


def fgsm_attack(input_batch_data: torch.Tensor, labels: torch.Tensor, epsilon: float, model: torch.nn.Module, loss: torch.nn.Module = torch.nn.CrossEntropyLoss(), device: str = "cpu") -> np.ndarray:
    """
    Generates adversarial example using fast gradient sign method.

    Args:
        input_batch_data (torch.Tensor): A list of input_data to be modified.
            EX: For 10 MNIST images, input_data.shape = (10, 1, 28, 28)
        labels (torch.Tensor): The true labels of the input data.
            EX: For 10 MNIST images, labels.shape = (10,)
        epsilon (float): Magnitude of the perturbation added to the input data.
        model (torch.nn.Module): The neural network model.
        loss (torch.nn.Module): The loss function to use for the computation.
        device (str): Device to use for the computation.
            default: "cpu"

    Returns:
        perturbed (np.ndarray): The perturbed input data.
    """
    # assert that epsilon is a positive value
    assert epsilon >= 0, "Epsilon must be a positive value"

    original_device = input_batch_data.device
    model.eval()

    # assert input data is a PyTorch tensor
    assert isinstance(input_batch_data, torch.Tensor), "Input data must be a PyTorch tensor"

    # assert that the input data and model are on the same device
    assert len(input_batch_data) == len(labels), "Input data and labels must have the same length"

    # assert epsilon is a positive value
    assert epsilon >= 0, "Epsilon must be a positive value"

    # assert that the input data and model are on the same device
    assert input_batch_data.device == next(model.parameters()).device, "Input data and model must be on the same device"

    new_input_data = []
    for input_data in input_batch_data:
        # Enable gradient computation for the input data
        input_data.requires_grad = True

        # Make a forward pass through the model and get the predicted class scores
        output = model(input_data.unsqueeze(0))

        # Calculate the loss between the predicted output and the true labels
        loss_value = loss(output, labels)

        # Zero the gradients
        model.zero_grad()

        # Perform backpropagation to compute the gradients
        loss_value.backward()

        # Get the gradients of the input data
        gradients = input_data.grad.data

        # Calculate the sign of the gradients
        sign_gradients = torch.sign(gradients)

        # Create the perturbation by scaling the sign of the gradients with epsilon
        perturbation = epsilon * sign_gradients

        # Add the perturbation to the input data to generate the adversarial example
        perturbed = input_data + perturbation

        # Move the perturbed data back to the original device
        perturbed = perturbed.to(original_device)

        # Return the perturbation as a numpy array
        new_input_data.append(perturbed.cpu().detach().numpy())

    return np.array(new_input_data)

