import Adversarial_Observation
from Adversarial_Observation.utils import seedEverything, buildCNN
from Adversarial_Observation.Attacks import *
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import os 

#============ cute visualizations===========
# define plot settings dictionary
plot_settings = {
    'font.size': 18,
    'xtick.major.size': 7,
    'xtick.major.width': 1.5,
    'ytick.major.size': 7,
    'ytick.major.width': 1.5,
    'xtick.minor.size': 4,
    'xtick.minor.width': 1,
    'ytick.minor.size': 4,
    'ytick.minor.width': 1,
    'axes.linewidth': 1.5,
    'legend.frameon': False,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True
}

# update plot settings
plt.rcParams.update(plot_settings)
#=========================================

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
    data, target = next(iter(test_loader))

    # add a batch dimension
    random = np.random.randint(0, len(data))
    img = data[random].unsqueeze(0)
    label = target[random]

    os.makedirs('FGSM', exist_ok=True)   
    # generate the attack
    for eps in epsilon:
        per = fgsm_attack(img, label, eps, model)
        

        plt.imshow(img.reshape(28,28), cmap='gray')
        condifence = model(img)[0].detach().numpy()[label]
        plt.title(f'Original\nConf.: {condifence:.4f}')
        plt.savefig(f'FGSM/original_eps_{eps}.png',bbox_inches='tight' )
        plt.clf()
        plt.close()
        
        plt.imshow(per.reshape(28,28), cmap='gray')
        condifence = model(img)[0].detach().numpy()[label]
        plt.title(f'Original\nConf.: {condifence:.4f}')
        plt.savefig(f'FGSM/Perterbation_eps_{eps}.png',bbox_inches='tight' )
        plt.clf()
        plt.close()
        
        # add the perturbation to the original image
        per = per + img

        plt.imshow(per.reshape(28,28), cmap='gray')
        condifence = model(torch.tensor(per).to(torch.float32).reshape(1,1, 28,28))[0].detach().numpy()[label]
        plt.title(f'Adversarial\nConf.: {condifence:.4f}')
        plt.savefig(f'FGSM/adver_eps_{eps}.png',bbox_inches='tight' )
        plt.clf()
        plt.close()

        act = activation_map(img, model)
        plt.imshow(act.reshape(28,28), cmap='gray')
        plt.title('Activation Map')
        plt.savefig(f'FGSM/adver_eps_{eps}_activation.png',bbox_inches='tight' )



if __name__ == '__main__':
    main()