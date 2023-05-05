import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def activation_map(input_data: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """
    Generate an activation map for an input data given a pre-trained PyTorch model.

    Args:
        input_data (torch.Tensor): Input data as a PyTorch tensor.
        model (torch.nn.Module): Pre-trained PyTorch model used to generate the activation map.

    Returns:
        activation_map (torch.Tensor): Activation map for the input data based on the input model.
    """
    # Set the model to evaluation mode and enable gradient computation for the input data
    model.eval()

    #  assert it has a batch dimension of 1 
    assert input_data.shape[0] == 1, "Input data must have a batch dimension of 1"
    
    # make sure the input data is on the same device as the model
    input_data.to(model.device)

    # Enable gradient computation for the input data
    input_data.requires_grad = True

    # Make a forward pass through the model and get the predicted class scores for the input data
    preds = model(input_data)

    # Get the index corresponding to the maximum predicted class score
    pred_class_idx = torch.argmax(preds)

    # Get the predicted class score corresponding to the maximum predicted class score
    score = preds[0, pred_class_idx]
    
    # Compute gradients of the score with respect to the input data pixels
    score.backward()

    # Compute the activation map as the absolute value of the gradients
    activation_map = input_data.grad

    # Return the activation map
    return activation_map

def fgsm_Attack(input_data: torch.Tensor, label: int, epsilon: float, model: torch.nn.Module, device: str = "cpu") -> torch.Tensor:
    """
    Generates adversarial example using fast gradient sign method.

    Args:
        input_data (torch.Tensor): The input_data to be modified.
        label (int): The true label of the input data.
        epsilon (float): Magnitude of the perturbation added to the input data.
        model (torch.nn.Module): The neural network model.
        device (str): Device to use for the computation. 
            default: "cpu"

    Returns:
        perturbed (torch.Tensor): the updated input with the noise added.
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

    # Create the perturbed data by adjusting each pixel of the input data
    with torch.no_grad():
        perturbed = data + epsilon * torch.sign(data.grad)

    # Return the perturbed data
    return perturbed.detach()
