import torch
import torchvision
import Adversarial_Observation as AO
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    # Seed everything
    AO.utils.seedEverything(6991)

    train_loader, test_loader = AO.utils.load_MNIST_data()
    model = AO.utils.load_MNIST_model()
    if os.path.isfile('MNIST_cnn.pt'):
        model.load_state_dict(torch.load('MNIST_cnn.pt'))
    else:
        raise Exception('MNIST_cnn.pt not found. Please run 1_build_train.py first')

    # Set model to evaluation mode
    model.eval()  
    
    #  get the first batch of the test data
    data, target = next(iter(test_loader))
    
    # get the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # move the data to the device
    data, target = data.to(device), target.to(device)
    model = model.to(device)

    fgsm(data, target, model, device)

def fgsm(imgs, labels, model, device):
    epsilons = [0, .1, .2, .3, .4, .5]
    # Run test for each epsilon
    for eps in epsilons:
        perterbed = AO.Attacks.fgsm_attack(imgs, model, (1,1,28,28), eps)
        plot_perterbed(perterbed, eps, imgs, labels, model)

def plot_perterbed(perterbed, eps, imgs, labels, model):
    for img, pert, label in zip(imgs, perterbed, labels):
        pert = torch.tensor(pert).to(torch.float32).to(img.device)
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img.cpu().numpy().squeeze(), cmap="gray")
        axs[0].set_title(f'Original')
        axs[1].imshow(pert.cpu().numpy().squeeze(), cmap="gray")
        axs[1].set_title(f'Perturbed:')
        axs[2].imshow((pert-img).cpu().numpy().squeeze(), cmap="gray")
        axs[2].set_title('Difference')
        plt.suptitle(f'Epsilon: {eps} Label: {label}')
        os.makedirs('./attack_results', exist_ok=True)
        plt.savefig(f'./attack_results/fgsm_{eps}_{label}.png')
        plt.close()
if __name__ == '__main__':
    main()
