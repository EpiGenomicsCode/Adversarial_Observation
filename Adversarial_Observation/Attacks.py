import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def activation_map(input_data: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """
    Generate an activation map for an input data given a pre-trained PyTorch model.

    Args:
        input_data (torch.Tensor): Input data as a PyTorch tensor. 
            EX. For 10 MNIST images, input_data.shape = (10, 1, 28, 28)
        model (torch.nn.Module): Pre-trained PyTorch model used to generate the activation map.

    Returns:
        activation_map (numpy.array): Activation map for the input data based on the input model.
    """
    # Set the model to evaluation mode and enable gradient computation for the input data
    model.eval()
    output_data = []
    # assert input data is a PyTorch tensor
    assert isinstance(input_data, torch.Tensor), "Input data must be a PyTorch tensor"
    
    # assert that the input data and model are on the same device
    assert input_data.device == model.parameters().__next__().device, "Input data and model must be on the same device"
    
    # Enable gradient computation for the input data
    input_data.requires_grad = True

    # Make a forward pass through the model and get the predicted class scores for the input data
    preds = model(input_data)

    for pred in preds:
        # Get the index corresponding to the maximum predicted class score
        pred_class_idx = torch.argmax(pred)

        # Get the predicted class score corresponding to the maximum predicted class score
        score = pred[pred_class_idx]

        # Compute gradients of the score with respect to the input data pixels
        score.backward()

        # Compute the activation map as the absolute value of the gradients
        activation_map = input_data.grad

        # save the activation map
        output_data.append(activation_map.detach().cpu().numpy())

    # Return the activation map
    return np.array(output_data)

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
        perturbed (np.ndarray): the noise to be added to the input data.
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

