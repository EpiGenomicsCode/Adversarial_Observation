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

    # Set model to evaluation mode
    model.eval()  

    # Create the adversarial image
    for data, target in train_loader:
        # for every image in the batch
        for i in range(len(data)):
            image, imagelabel = data[i], target[i]
            # if it is not misclassified
            if torch.argmax(model(image.unsqueeze(0))) == imagelabel:
                # attack it
                attack_image(image, model, imagelabel)
       
def attack_image(image, model, label):
        # if the folder already exists then skip
        original_confidence = model(image.unsqueeze(0))[0]
        for eps in [.05,.01, .1, .2, .3, .5][::-1]:
            if not os.path.isfile(f'Noise/fgsm/{label.item()}/{eps}.png'):   
                per = AO.Attacks.fgsm_attack(image.unsqueeze(0), label.unsqueeze(0), eps, model,
                                            loss=torch.nn.CrossEntropyLoss())
                per = per[0]
                
                noise = per - image.detach().numpy()
                per = np.clip(per, 0, 1)
                noise = noise[0] # 28,28
                per = per[0] # 28,28
                activation = AO.activation_map(torch.tensor(per).unsqueeze(0).unsqueeze(0), model)
                confidence =model(torch.tensor(per).unsqueeze(0).unsqueeze(0))[0]
                new_label = torch.argmax(confidence)
                if new_label == label:
                    continue
                else:
                    os.makedirs(f'Noise/fgsm/{label.item()}', exist_ok=True)
                    score = confidence[new_label]
                    score = score.detach().numpy().round(3)

                    #  plot as a 3x1 image
                    fig, axs = plt.subplots(1, 3, figsize=(10, 10))
                    axs[0].imshow(image[0], cmap='gray')
                    axs[0].axis('off')

                    axs[1].imshow(noise, cmap='gray')
                    axs[1].set_title(f'Label: {label.item()}, New Label: {new_label.item()}, Score: {score.item()}')
                    axs[1].axis('off')

                    axs[2].imshow(per, cmap='gray')
                    axs[2].axis('off')
                    plt.savefig(f'Noise/fgsm/{label.item()}/{eps}.png', bbox_inches='tight', pad_inches=0)
                    # save all 3 images for reference
                    plt.clf()
                    np.save(f'Noise/fgsm/{label.item()}/{eps}.npy', noise)
                    np.save(f'Noise/fgsm/{label.item()}/{eps}_perturbed.npy', per)
                    np.save(f'Noise/fgsm/{label.item()}/{eps}_activation.npy', activation)
                    np.save(f'Noise/fgsm/{label.item()}/{eps}_original.npy', image.detach().numpy())
                    plt.close()
                    plt.clf()                  

if __name__ == '__main__':
    main()
