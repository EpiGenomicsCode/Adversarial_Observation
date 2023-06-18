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
        #  for each input in the batch, compute the gradient of the output with respect to the input
        gradients = torch.autograd.grad(outputs=output, inputs=input_data, grad_outputs=torch.ones_like(output), create_graph=True)[0]

    
    activation_map = gradients.detach().cpu().numpy()  # Convert the activation map to a numpy array

    if normalize:
        min_vals = np.min(activation_map, axis=(1, 2), keepdims=True)
        max_vals = np.max(activation_map, axis=(1, 2), keepdims=True)
        activation_map = (activation_map - min_vals) / (max_vals - min_vals + 1e-8)


    return activation_map


def fgsm_attack(input_batch_data: torch.Tensor, labels: torch.Tensor, epsilon: float, model: torch.nn.Module, loss: torch.nn.Module = torch.nn.CrossEntropyLoss(), device: str = "cpu") -> np.ndarray:
    """
    Generates adversarial examples using the Fast Gradient Sign Method (FGSM).

    Args:
        input_batch_data (torch.Tensor): Input data to be modified. Shape: (batch_size, channels, height, width)
        labels (torch.Tensor): The true labels of the input data. Shape: (batch_size,)
        epsilon (float): Magnitude of the perturbation added to the input data.
        model (torch.nn.Module): The neural network model.
        loss (torch.nn.Module): The loss function to use for the computation. Default: torch.nn.CrossEntropyLoss()
        device (str): Device to use for the computation. Default: "cpu"

    Returns:
        np.ndarray: The perturbed input data.
    """
    assert epsilon >= 0, "Epsilon must be a non-negative value"
    assert isinstance(input_batch_data, torch.Tensor), "Input data must be a PyTorch tensor"
    assert len(input_batch_data) == len(labels), "Input data and labels must have the same length"
    assert input_batch_data.device == next(model.parameters()).device, "Input data and model must be on the same device"

    model.eval()
    original_device = input_batch_data.device

    new_input_data = []
    for input_data, label in zip(input_batch_data, labels):
        input_data = input_data.to(device)
        input_data.requires_grad = True

        output = model(input_data.unsqueeze(0))
        loss_value = loss(output, label.unsqueeze(0))

        model.zero_grad()
        loss_value.backward()

        gradients = input_data.grad.data
        sign_gradients = torch.sign(gradients)

        perturbation = epsilon * sign_gradients
        perturbed = input_data + perturbation

        perturbed = perturbed.to(original_device)

        new_input_data.append(perturbed.cpu().detach().numpy())

    return np.array(new_input_data)


