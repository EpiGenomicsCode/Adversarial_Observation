import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def activation_map(input_data: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """
    Generate an activation map for an input image given a pre-trained PyTorch model.

    Args:
        input_data (torch.Tensor): Input image as a PyTorch tensor.
        model (torch.nn.Module): Pre-trained PyTorch model used to generate the activation map.

    Returns:
        activation_map (torch.Tensor): Activation map for the input image.
    """
    # Set the model to evaluation mode and enable gradient computation for the input image
    model.eval()
    input_data.requires_grad = True

    # Disable gradient computation for all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Make a forward pass through the model and get the predicted class scores for the input image
    preds = model(input_data)

    # Compute the score
    score = preds[0, torch.argmax(preds)].to(torch.float32)

    # Compute gradients of the score with respect to the input image pixels
    score.backward()

    # Compute the activation map as the absolute value of the gradients
    activation_map = input_data.grad

    # Return the activation map
    return activation_map

def fgsm_attack(input_data: torch.Tensor, label: int, epsilon: float, model: torch.nn.Module, device: str = "cpu") -> torch.Tensor:
    """
    Generates adversarial example using fast gradient sign method.

    Args:
        input_data (torch.Tensor): The input_data to be modified.
        label (int): The true label of the input image.
        epsilon (float): Magnitude of the perturbation added to the input image.
        model (torch.nn.Module): The neural network model.
        device (str): Device to use for the computation. Defaults to "cpu".

    Returns:
        The modified image tensor.
    """
    model.eval()
    data = input_data.to(device)
    data.requires_grad = True

    # Forward pass to get the prediction
    output = model(data)

    # Calculate the loss
    loss = F.cross_entropy(output, torch.tensor([label]).to(device))

    # Backward pass to get the gradient
    model.zero_grad()
    loss.backward()

    # Create the perturbed image by adjusting each pixel of the input image
    with torch.no_grad():
        perturbed = data + epsilon * torch.sign(data.grad)
        perturbed = torch.clamp(perturbed, 0, 1)

    # Return the perturbed image
    return perturbed.detach()
