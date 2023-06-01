import torch
import torchvision
import Adversarial_Observation as AO
import numpy as np
import matplotlib.pyplot as plt
import os
from util import *


def seedEverything(seed):
    """
    Seeds all the random number generators to ensure reproducibility.

    Args:
        seed: The seed value for random number generators.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    # Seed everything
    seedEverything(42)

    train_loader, test_loader = load_MNIST_data()
    model = build_MNIST_Model()
    model.load_state_dict(torch.load('MNIST_cnn.pt'))
    shape = (1, 1, 28, 28)

    # Set model to evaluation mode
    model.eval()  

    data, target = next(iter(test_loader))

    # Get the first image and label
    image = torch.clamp(data[0], 0, 1)
    label = target[0]

    os.makedirs('Noise', exist_ok=True)
    plt.imshow(image[0], cmap='gray')
    pred = model(image.unsqueeze(0))
    confidence = torch.argmax(pred)
    confidence_score = pred[0][confidence]
    plt.title(f"Original\nconfidence {confidence_score}")
    plt.savefig('Noise/original.png')

    # Plot the activation

    activation = AO.Attacks.activation_map(image.reshape(shape), model)
    plt.imshow(activation[0][0].reshape(28,28), cmap='gray')
    plt.colorbar()
    plt.title("Activation Map")
    plt.savefig('Noise/activation.png')
    plt.clf()

    os.makedirs('Noise/fgsm', exist_ok=True)
    os.makedirs('Noise/fgsm_noise', exist_ok=True)
    os.makedirs('Noise/fgsm_activation', exist_ok=True)

    # Create the adversarial image
    for eps in [.01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
        per = AO.Attacks.fgsm_attack(image.unsqueeze(0), label.unsqueeze(0), eps, model,
                                    loss=torch.nn.CrossEntropyLoss())
        per = torch.clamp(torch.tensor(per), 0, 1)

        # Plot the adversarial image
        plt.imshow(per[0][0], cmap='gray')
        pred = model(per)
        plt.colorbar()
        plt.savefig(f'Noise/fgsm/{eps}.png')
        plt.clf()

        # Plot the noise
        noise = per[0][0] - image[0] 
        noise = noise/eps
        plt.imshow(noise, cmap='gray')
        plt.colorbar()
        plt.savefig(f'Noise/fgsm_noise/{eps}.png')
        plt.clf()

        # Plot the activation
        activation = AO.Attacks.activation_map(per, model)
        plt.imshow(activation[0][0], cmap='gray')
        plt.colorbar()
        plt.title("Activation Map")
        plt.savefig(f'Noise/fgsm_activation/{eps}.png')
        plt.clf()


if __name__ == '__main__':
    main()
