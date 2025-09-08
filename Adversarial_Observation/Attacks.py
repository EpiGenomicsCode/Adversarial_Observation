import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def fgsm_attack(input_batch_data: torch.Tensor, model: torch.nn.Module, input_shape: tuple, epsilon: float) -> torch.Tensor:
    """
    Apply the FGSM attack to input images given a pre-trained PyTorch model.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_batch_data = input_batch_data.to(device)

    adversarial_batch_data = []

    for img in input_batch_data:
        # Make a copy of the image and enable gradient tracking
        img = img.clone().detach().unsqueeze(0).to(device)
        img.requires_grad = True

        # Forward pass
        preds = model(img)
        target = torch.argmax(preds, dim=1)
        loss = F.cross_entropy(preds, target)

        # Backward pass
        model.zero_grad()
        loss.backward()

        # Generate perturbation
        grad = img.grad.data
        adversarial_img = img + epsilon * grad.sign()
        adversarial_img = torch.clamp(adversarial_img, 0, 1)

        adversarial_batch_data.append(adversarial_img.squeeze(0).detach())

    return torch.stack(adversarial_batch_data)

def compute_gradients(model, img, target_class):
    preds = model(img)
    target_score = preds[0, target_class]
    return torch.autograd.grad(target_score, img)[0]

def generate_adversarial_examples(input_batch_data, model, method='fgsm', **kwargs):
    if method == 'fgsm':
        return fgsm_attack(input_batch_data, model, **kwargs)
    # Implement other attack methods as needed

def gradient_ascent(input_image, model, input_shape, target_class, num_iterations=100, step_size=0.01):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_image = input_image.to(device).detach().requires_grad_(True)

    for _ in range(num_iterations):
        gradients = compute_gradients(model, input_image.reshape(input_shape), target_class)
        input_image = input_image + step_size * gradients.sign()
        input_image = torch.clamp(input_image, 0, 1)
        input_image = input_image.detach().requires_grad_(True)

    return input_image.cpu().detach().numpy()

def gradient_map(input_image, model, input_shape):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_image = torch.tensor(input_image).to(device).detach().requires_grad_(True)

    preds = model(input_image.reshape(input_shape))
    target_class = torch.argmax(preds)
    loss = F.cross_entropy(preds, target_class.unsqueeze(0))

    model.zero_grad()
    loss.backward()

    gradient = input_image.grad.data.cpu().numpy()
    gradient = np.abs(gradient).mean(axis=1)  # Average over channels if needed
    return gradient
    
def visualize_adversarial_examples(original, adversarial):
    # Code to visualize original vs adversarial images
    pass

def log_metrics(success_rate, average_perturbation):
    logging.info(f'Success Rate: {success_rate}, Average Perturbation: {average_perturbation}')

class Config:
    def __init__(self, epsilon=0.1, attack_method='fgsm'):
        self.epsilon = epsilon
        self.attack_method = attack_method