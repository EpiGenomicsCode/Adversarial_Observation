# Import statements
import os
from typing import Tuple
from typing import List
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from Adversarial_Observation.utils import *
from Adversarial_Observation.Swarm_Observer.Swarm import PSO
from Adversarial_Observation.visualize import * 
from sklearn.decomposition import PCA
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
import pickle

# ==== Global Variables ====
# The value of the column we want to optimize for
endValue: int = 1
# The value of the column we want to start with
startValue: int = 2
#================================================

def costFunc(model: torch.nn.Module, input: torch.Tensor) -> np.ndarray:
    """
    This function takes a model and a tensor input, reshapes the tensor to be a 28 x 28 image of batch size 1 and passes that through
    the model. Then, it returns the output of the column of interest (endValue) as a numpy array.
    """
    input = torch.reshape(input, (1, 1, 28, 28)).to(torch.float32)
    out = model(input)
    return out.detach().numpy()[0][endValue]


def main() -> None:
    """
    The main function that seeds the model, builds a model with the same architecture as the one you want to attack,
    loads the weights of the model you want to attack, attacks the model with your designed attack, and visualizes
    the results.
    """
    seedEverything(44)
    # Step 1: build a model with the same architecture as the one you want to attack
    model = buildCNN(10)

    # Step 2: load the weights of the model you want to attack
    model.load_state_dict(torch.load('./saved_models/MNIST_CNN.pt'))
    model.eval()

    # Step 3: attack the model with your designed attack
    epochs = 100
    attack_model_apso_targeted(model, epochs, outdir='./onesInit/')



def attack_model_apso_targeted(model: torch.nn.Module, epochs: int, outdir: str = "./Output/") -> None:
    """
    This function will take a model and will randomly generate inputs to the model and attack those inputs
    :param model: the model you want to attack
    :param epochs: the number of iterations for the attack
    :param outdir: the directory you want to save the results to
    """

    # create the output directory
    os.makedirs('.{}apso'.format(outdir), exist_ok=True)

    # Step 4: Generate the parameters for inputs to be attacked
    # Step 4.5: Download the MNIST dataset (we are attacking this dataset)
    mnist = torchvision.datasets.MNIST('./data', 
                                       train=True, 
                                       download=True, 
                                       transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))
                                                          ]
                                                          )
                                        )
    
    # Step 4.75: Create a dataloader to load the images from the MNIST dataset
    dataloader = DataLoader(mnist, batch_size=1, shuffle=False)

    # Step 5: Get the inputs to be attacked
    images = np.array([i[0].detach().cpu().numpy().flatten() for i in dataloader if i[1] == startValue]).reshape(-1, 1, 28, 28)


    # (Optional): Used for Visualizing the swarm in PCA space
    pca = pickle.load(open('./Create/artifacts/pca.pkl', 'rb'))

    # Step 6: Run the attack
    attackLoop(images, model, epochs, outdir, pca=pca)

    # (Optional): Visualize the results
    visualizeGIF(f"{outdir}apso/images/best", ["epoch_{}.png".format(i) for i in range(1, epochs)], "best.gif")
    visualizeGIF(f"{outdir}apso/images/average", ["epoch_{}.png".format(i) for i in range(1, epochs)], "average.gif")
    visualizeGIF(f"{outdir}apso/images/PCA", ["epoch_{}.png".format(i) for i in range(0,1+epochs)], "PCA.gif")

def attackLoop(inputData, model, epochs, outdir, pca=None):
    # Initialize the swarm
    apso = PSO(inputData, costFunc, model, w=.8, c1=.5, c2=.5)

    # Fancy progress bar
    loop = tqdm.tqdm(range(1, epochs+1))
    
    # (Optional): Plot the initial swarm
    PCASwarm(pca, [p.position_i for p in apso.swarm], 0, outdir)
    
    # Loop through the attack
    for i in loop:
        # Step the swarm
        positions, best_positions = apso.step()
        #==================Logging/Visualization===============================
            # Get the positions to fit the model
        positions = cleanPositions(positions)

            # Get the highest confidence prediction and the predictions of the swarm
        best_score, newpredictions = bestInSwarm(model, positions, i, endValue,outdir)

            # Sum the positions of the swarm and get its confidence prediction
        whole_label = convergeOfSwarm(model, positions, i, outdir)

            # calculate the percent of particles the converged to the endValue
        per = np.sum(newpredictions[:, endValue] > .5)/len(newpredictions)

            # Plot the swarm in PCA space
        if pca != None:
            PCASwarm(pca, positions,i, outdir)

        #================End Logging/Visualization===============================
        
        # Update the progress bar
        loop.set_description(f"APSO best score: {best_score:.2f}, Sum predicted label: {whole_label}, Percent of particles labeled as {endValue}: {per:.2f}")


def PCASwarm(pca: PCA, position: List[np.ndarray], epoch: int, outdir: str) -> None:
    """
    This function generates a PCA plot of the positions and saves it to the specified output directory.
    
    :param pca: A PCA object to transform the positions.
    :param position: A list of position numpy arrays to plot.
    :param epoch: The current epoch number for the plot title.
    :param outdir: The output directory to save the plot to.
    """
    # Transform the positions with the PCA object
    reducedPositions = pca.transform(np.array([i[0].detach().cpu().numpy().flatten() for i in position]))

    # Load the transform and labels
    transform = np.load('./Create/artifacts/transform.npy')
    labels = np.load('./Create/artifacts/labels.npy')

    # Clear the current figure
    plt.clf()

    # Set the figure size
    plt.figure(figsize=(10, 10))

    # Plot the transform with the labels as the color
    labels =[]
    for label in np.unique(labels):
        if label == startValue or label == endValue:
            plt.scatter(transform[labels == label, 0], transform[labels == label, 1], s=10, label=label)
            labels.append(label)

    # Plot the legend for the labels
    plt.legend(labels, loc='upper right', title='Labels')

    # Plot the swarm as black x markers
    plt.scatter(reducedPositions[:, 0], reducedPositions[:, 1], c='k', marker='x', s=50)

    # Set the plot title
    plt.title(f"Epoch {epoch}")

    # Check if the directory exists, and create it if it doesn't
    os.makedirs('{}apso/images/PCA'.format(outdir), exist_ok=True)

    # Save the plot to the specified output directory
    plt.savefig(f"{outdir}apso/images/PCA/epoch_{epoch}.png")

    # Close the current figure
    plt.close()

    # Clear the current figure
    plt.clf()

if __name__ == '__main__':
    main()