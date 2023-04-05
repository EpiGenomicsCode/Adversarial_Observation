import Adversarial_Observation
from Adversarial_Observation.utils import seedEverything, buildCNN
from Adversarial_Observation.Attacks import fgsm_attack, saliency_map
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os 

# load the data
def loadData():
    """
    Load the MNIST dataset from torchvision.datasets
    :return: train_loader, test_loader
    """
    return  (
            torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(
                    './data',
                    train=True,
                    download=True,
                    transform=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            (0.1307,), (0.3081,))
                    ])),
                batch_size=64,
                shuffle=True),

            torch.utils.data.DataLoader(
                torchvision.datasets.MNIST(
                    './data',
                    train=False,
                    download=True,
                    transform=torchvision.transforms.Compose([
                        torchvision.transforms.ToTensor(),
                        torchvision.transforms.Normalize(
                            (0.1307,), (0.3081,))
                    ])),
                batch_size=1000,
                shuffle=True)
            )

def main():
    seedEverything(44)
    # load the data
    train_loader, test_loader = loadData()

    # load the model
    model = buildCNN(10)

    # load the weights
    model.load_state_dict(torch.load('./artifacts/mnist_cnn.pt'))

    epsilon = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]

    # get one image to attack from the test loader
    for data, target in test_loader:
        break

    # add a batch dimension
    img = data[0].unsqueeze(0)
    label = target[0]

    os.makedirs('FGSM', exist_ok=True)   
    # generate the attack
    for eps in epsilon:
        per = fgsm_attack(img, label, eps, model)
        # create a 1x2 subplot where the first image is the original image and the second is the perturbed image
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img.reshape(28,28), cmap='gray')
        ax[0].set_title('Original Image')

        attacked = per + img.numpy()
        attacked = attacked.reshape(28, 28)

        ax[1].imshow(attacked, cmap='gray')
        ax[1].set_title('Perturbed Image')
        plt.savefig(f'FGSM/eps_{eps}.png')

    os.makedirs('Activation', exist_ok=True)
    sal = saliency_map(img, model)

    # create a 1x2 subplot where the first image is the original image and the second is the activation map
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img.reshape(28,28), cmap='gray')
    ax[0].set_title('Original Image')
    ax[1].imshow(sal.reshape(28,28), cmap='gray')
    ax[1].set_title('Activation Map')
    plt.savefig('Activation/activation.png')


if __name__ == '__main__':
    main()