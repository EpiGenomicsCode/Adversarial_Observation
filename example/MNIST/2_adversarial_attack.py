import torch
import torchvision
import Adversarial_Observation as AO
import numpy as np
import matplotlib.pyplot as plt
import os
import tqdm 

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

<<<<<<< HEAD
    # grad_ascent(model, device)
    fgsm(data, target, model, device)

def grad_ascent(model, device):
    #  get the gradient ascent
    os.makedirs('./gradient_Ascent', exist_ok=True)
    for i in tqdm.tqdm(range(10), desc='Gradient Ascent'):
        ga = AO.Attacks.gradient_ascent(torch.tensor(np.random.random((1,1,28,28))), model, (1,1,28,28), i, 1000, 10)
        plt.imshow(ga[0].reshape(28,28), cmap='jet')
        plt.colorbar()
        confidence =  model(torch.tensor(ga[0].reshape(1,1,28,28)).to(device))[0][i].item()
        plt.title(f"Gradient Ascent: label {i}, confidence {confidence}")
        plt.savefig(f'./gradient_Ascent/ga_{i}.png')
        plt.close()

def fgsm(imgs, labels, model, device):
    epsilons = [.001, .01, .1, .2]
    # Run test for each epsilon
    for eps in epsilons:
        perterbed = AO.Attacks.fgsm_attack(imgs, model, (1,1,28,28), eps)
        plot_perterbed(perterbed, eps, imgs, labels, model)

def plot_perterbed(perterbed, eps, imgs, labels, model):
    for img, pert, label in zip(imgs, perterbed, labels):
        pert = torch.tensor(pert).to(torch.float32).to(img.device)
        per_conf = round(model(pert.unsqueeze(0))[0][label].item(), 2)

        if per_conf < .9:
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(img.cpu().numpy().squeeze(), cmap="gray")
            axs[0].set_title(f'Original')
            axs[1].imshow(pert.cpu().numpy().squeeze(), cmap="gray")
            axs[1].set_title(f'Perturbed Confidence: {per_conf}')
            axs[2].imshow((pert-img).cpu().numpy().squeeze(), cmap="gray")
            axs[2].set_title('Noise')
            plt.suptitle(f'Epsilon: {eps} Label: {label}')
            os.makedirs('./attack_results', exist_ok=True)
            plt.savefig(f'./attack_results/fgsm_{eps}_{label}.png')
            # save perturbed image as npy
            np.save(f'./attack_results/fgsm_{eps}_{label}.npy', pert.cpu().numpy().squeeze())
            plt.close()
            #  make a 2x1 plot of the activation of the original and perturbed image
            fig, axs = plt.subplots(1, 2, figsize=(12, 4))
            axs[0].imshow( AO.Attacks.gradient_map(img.reshape(1,1,28,28), model, (1,1,28,28))[0], cmap="jet")
            axs[0].set_title(f'Original Confidence: {1}')
            axs[1].imshow( AO.Attacks.gradient_map(pert.reshape(1,1,28,28), model, (1,1,28,28))[0], cmap="jet")
            axs[1].set_title(f'Perturbed')
            plt.suptitle(f'Epsilon: {eps} Label: {label}')
            plt.savefig(f'./attack_results/fgsm_{eps}_{label}_grad.png')
            plt.close()


=======
    grad_ascent(model, device)
    fgsm(data, target, model, device)

def grad_ascent(model, device):
    #  get the gradient ascent
    os.makedirs('./gradient_Ascent', exist_ok=True)
    for i in tqdm.tqdm(range(10), desc='Gradient Ascent'):
        ga = AO.Attacks.gradient_ascent(torch.tensor(np.random.random((1,1,28,28))), model, (1,1,28,28), i, 1000, 10)
        plt.imshow(ga[0].reshape(28,28), cmap='jet')
        plt.colorbar()
        confidence =  model(torch.tensor(ga[0].reshape(1,1,28,28)).to(device))[0][i].item()
        plt.title(f"Gradient Ascent: label {i}, confidence {confidence}")
        plt.savefig(f'./gradient_Ascent/ga_{i}.png')
        plt.close()

def fgsm(imgs, labels, model, device):
    epsilons = [0, .1, .2, .3, .4, .5]
    # Run test for each epsilon
    for eps in epsilons:
        perterbed = AO.Attacks.fgsm_attack(imgs, model, (1,1,28,28), eps)
        plot_perterbed(perterbed, eps, imgs, labels, model)
>>>>>>> 8f0834909dfca238b88ce6296712862ccd07bea7

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
