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

    train_loader, test_loader = load_CIFAR10_data()
    model = build_CIFAR10_Model()
    model.load_state_dict(torch.load('CIFAR10_cnn.pt'))
    shape = (1, 3, 32, 32)

    # Set model to evaluation mode
    model.eval()  

    data, target = next(iter(test_loader))

    # Get the first image and label
    image = torch.clamp(data[0], 0, 1)
    label = target[0]

    os.makedirs('Noise', exist_ok=True)
    plt.imshow(image[0])
    pred = model(image.unsqueeze(0))
    confidence = torch.argmax(pred)
    confidence_score = pred[0][confidence]
    plt.title(f"Original\nconfidence {confidence_score}")
    plt.savefig('Noise/original.png')

    # Plot the activation

    activation = AO.Attacks.activation_map(image.reshape(shape), model)
    plt.imshow(activation[0][0])
    plt.colorbar()
    plt.title("Activation Map")
    plt.savefig('Noise/activation.png')
    plt.clf()

    os.makedirs('Noise/fgsm', exist_ok=True)
    os.makedirs('Noise/fgsm_noise', exist_ok=True)
    os.makedirs('Noise/fgsm_activation', exist_ok=True)

    # Create the adversarial image
    for eps in [0, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
        per = AO.Attacks.fgsm_attack(image.reshape(shape), [label], eps, model,
                                    loss=torch.nn.CrossEntropyLoss())
        per = np.clip(per, 0, 1)
        per = torch.tensor(per)
        # Plot the adversarial image
        plt.imshow(per[0][0])
        pred = model(per)
        confidence = torch.argmax(pred)
        confidence_score = pred[0][confidence]
        plt.colorbar()
        plt.title(f"FGSM\nconfidence {confidence_score}")
        plt.savefig(f'Noise/fgsm/{eps}.png')
        plt.clf()

        # Plot the noise
        plt.imshow(torch.clamp(per[0][0] - image[0], 0, 1))
        plt.colorbar()
        plt.title(f"FGSM Noise\nconfidence {confidence_score}")
        plt.savefig(f'Noise/fgsm_noise/{eps}.png')
        plt.clf()

        # Plot the activation
        activation = AO.Attacks.activation_map(per, model)
        plt.imshow(activation[0][0])
        plt.colorbar()
        plt.title("Activation Map")
        plt.savefig(f'Noise/fgsm_activation/{eps}.png')
        plt.clf()


if __name__ == '__main__':
    main()
