import torch
import torchvision
import Adversarial_Observation as AO
import numpy as np
import matplotlib.pyplot as plt
import os
from util import *

labelDic ={ 0: 'airplane',
            1: 'automobile',
            2: 'bird',
            3: 'cat',
            4: 'deer',
            5: 'dog',
            6: 'frog',
            7: 'horse',
            8: 'ship',
            9: 'truck'
            }
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
    datas, targets = next(iter(test_loader))

    for image, label in zip(datas, targets):
        labelName = labelDic[label.item()]
        os.makedirs('Noise', exist_ok=True)
        #  if the length of the subdirectory is 9 break
        if len(os.listdir(f'Noise/')) >= 9:
            break

       
        os.makedirs(f'Noise/{labelName}', exist_ok=True)
        plt.imshow(image.numpy().transpose(1,2,0))
        confidence_score = label
        plt.title(f"Original\nconfidence {confidence_score}")

        plt.savefig(f'Noise/{labelName}/original.png')


        # Plot the activation

        activation = AO.Attacks.activation_map(image.reshape(shape), model)
        plt.imshow(activation[0].transpose(1,2,0))
        plt.colorbar()
        plt.title("Activation Map")
        plt.savefig(f'Noise/{labelName}/activation.png')
        plt.clf()

        os.makedirs(f'Noise/{labelName}/fgsm', exist_ok=True)
        os.makedirs(f'Noise/{labelName}/fgsm_noise', exist_ok=True)
        os.makedirs(f'Noise/{labelName}/fgsm_activation', exist_ok=True)

        # Create the adversarial image
        for eps in [0, .01, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9]:
            per = AO.Attacks.fgsm_attack(torch.tensor(image).unsqueeze(0), [label], eps, model)
            
            # Plot the adversarial image
            plt.imshow(per[0].transpose(1,2,0)/per[0].max())
            pred =  model(torch.tensor(per[0]).unsqueeze(0))[0]
            confidence = torch.argmax(pred)
            
            confidence_score = pred[confidence]
            plt.colorbar()
            plt.title(f"FGSM\nconfidence {confidence_score}")
            plt.savefig(f'Noise/{labelName}/fgsm/{eps}.png')
            plt.clf()

            # Plot the noise
            noise = per[0] - image.numpy()
            plt.imshow(noise.transpose(1,2,0)/noise.max())
            plt.colorbar()
            plt.savefig(f'Noise/{labelName}/fgsm_noise/{eps}.png')
            plt.clf()

            # Plot the activation
            activation = AO.Attacks.activation_map(torch.tensor(per[0]).unsqueeze(0), model)
            plt.imshow(activation[0].transpose(1,2,0)/activation[0].max(), cmap="jet")
            plt.colorbar()
            plt.title("Activation Map")
            plt.savefig(f'Noise/{labelName}/fgsm_activation/{eps}.png')
            plt.clf()

            #  save the adversarial image as numpy array
            np.save(f'Noise/{labelName}/fgsm/{eps}.npy', per[0].numpy())
            np.save(f'Noise/{labelName}/fgsm_noise/{eps}.npy', noise)



if __name__ == '__main__':
    main()
