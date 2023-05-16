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

    for image, label in zip(data, target):
    
        os.makedirs('Noise', exist_ok=True)
        os.makedirs(f'Noise/{label}', exist_ok=True)
        #  if the length of the subdirectory is 9 break
        if len(os.listdir(f'Noise/')) > 9:
            break
        
        plt.imshow(image[0], cmap='gray')
        pred = model(image.unsqueeze(0))
        confidence = torch.argmax(pred)
        confidence_score = pred[0][confidence]
        plt.title(f"Original\nconfidence {confidence_score}")
        plt.savefig(f'Noise/{label}/original.png')

        # Plot the activation

        activation = AO.Attacks.activation_map(image.reshape(shape), model)
        plt.imshow(activation[0][0], cmap='jet')
        plt.colorbar()
        plt.title("Activation Map")
        plt.savefig(f'Noise/{label}/activation.png')
        plt.clf()

        os.makedirs(f'Noise/{label}/fgsm', exist_ok=True)
        os.makedirs(f'Noise/{label}/fgsm_noise', exist_ok=True)
        os.makedirs(f'Noise/{label}/fgsm_activation', exist_ok=True)

        # Create the adversarial image
        for eps in [0, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
            per = AO.Attacks.fgsm_attack(image.reshape(shape), [label], eps, model,
                                        loss=torch.nn.CrossEntropyLoss())
            per = np.clip(per, 0, 1)
            per = torch.tensor(per)
            # Plot the adversarial image
            plt.imshow(per[0][0], cmap='gray')
            pred = model(per)[0]
            confidence = torch.argmax(pred)
            confidence_score = pred[confidence]
            plt.colorbar()
            plt.title(f"FGSM\nconfidence {confidence_score}")
            plt.savefig(f'Noise/{label}/fgsm/{eps}.png')
            plt.clf()

            # Plot the noise
            plt.imshow(torch.clamp(per[0][0] - image[0], 0, 1), cmap='gray')
            plt.colorbar()
            plt.title(f"FGSM Noise\nconfidence {confidence_score}")
            plt.savefig(f'Noise/{label}/fgsm_noise/{eps}.png')
            plt.clf()

            # Plot the activation
            activation = AO.Attacks.activation_map(per, model)
            plt.imshow(activation[0][0], cmap='jet')
            plt.colorbar()
            plt.title("Activation Map")
            plt.savefig(f'Noise/{label}/fgsm_activation/{eps}.png')
            plt.clf()

            # save the adversarial image and noise as a numpy array
            np.save(f'Noise/{label}/fgsm/{eps}.npy', per[0][0])
            np.save(f'Noise/{label}/fgsm_noise/{eps}.npy', torch.clamp(per[0][0] - image[0], 0, 1))
            



if __name__ == '__main__':
    main()
