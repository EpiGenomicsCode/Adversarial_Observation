import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def saliency_map(img: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """
    Generate a saliency map for an input image given a pre-trained PyTorch model.

    Args:
        img (ndarray): Input image as a 3D numpy array.
        model (nn.Module): Pre-trained PyTorch model used to generate the saliency map.

    Returns:
        saliency map (ndarray): Saliency map for the input image.
    """
    # Convert the input image to a PyTorch tensor with dtype=torch.float32 and enable gradient computation
    img = img.clone().detach().requires_grad_(True)
    model.eval()

    # Disable gradient computation for all model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Make a forward pass through the model and get the predicted class scores for the input image
    preds = model(img)

    # Compute the score and index of the class with the highest predicted score
    score, _ = torch.max(preds, 1)

    # Compute gradients of the score with respect to the input image pixels
    score.backward()

    # Compute the saliency map by taking the maximum absolute gradient across color channels and normalize the values
    slc = torch.max(torch.abs(img.grad), dim=0)[0]
    slc = (slc - slc.min()) / (slc.max() - slc.min())

    # return the saliency map as a numpy array
    return slc.detach().numpy()


def fgsm_attack(input_data: torch.Tensor, label: int, epsilon: float, model: torch.nn.Module, device: str ="cpu") -> torch.Tensor:
    """
    Generates adversarial example using fast gradient sign method.

    Args:
        input_data (torch.Tensor): The input_data to be modified.
        label (int): The true label of the input image.
        epsilon (float): Magnitude of the perturbation added to the input image.
        model (torch.nn.Module): The neural network model.

    Returns:
        The modified image tensor.
    """
 
    data = torch.tensor(input_data).to(torch.float32).to(device)
    model = model.to(device)
    data.requires_grad = True
   
    # Forward pass to get the prediction
    output = model(data)

    # Calculate the loss
    loss = F.cross_entropy(output, torch.tensor([label]).to(device))

    # Backward pass to get the gradient
    loss.backward()
    model.zero_grad()


    # Create the perturbed image by adjusting each pixel of the input image
    perturbed = data + epsilon * data.grad.sign() 
    perturbed = torch.clamp(perturbed, 0, 1)
    
    # Return the perturbed image
    return perturbed.cpu().detach().numpy()
