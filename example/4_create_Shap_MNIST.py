import torch
import torchvision
import shap
import numpy as np
import matplotlib.pyplot as plt
from util import *
import os
dataset = "MNIST"


import os
import matplotlib.pyplot as plt
import torch
import shap


def save_and_plot_shap_values(dataloader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # get the first batch of data
    data, target = next(iter(dataloader))

    # move data and target to device
    data = data.to(device)
    target = target.to(device)

    # get only the first 10 samples
    data = data[:10]
    target = target[:10]

    # move model to device
    model = model.to(device)

    # generate SHAP values
    explainer = shap.DeepExplainer(model, data)
    shap_values = explainer.shap_values(data)

    os.makedirs('SHAP', exist_ok=True)

    # create a 10x10 grid of subplots
    fig, axes = plt.subplots(10, 11, figsize=(100, 100))

    # iterate over the SHAP values and plot on the subplots
    for i in range(len(data)):
        shap_i = shap_values[i]
        label = target[i]

        # plot the original image
        axes[i, 0].imshow(data[i].cpu().reshape(28, 28), cmap='gray')
        axes[i, 0].set_title(f'Label: {label}')

        # plot the SHAP values
        num_shap_values = min(10, len(shap_i))  # Adjust the number of SHAP values to fit within the grid
        for j in range(num_shap_values):
            axes[i, j+1].imshow(shap_i[j].reshape(28, 28), cmap='jet')
            axes[i, j+1].axis('off')
            axes[i, j+1].set_title(f'SHAP value: {j}')

    # remove empty rows
    for i in range(len(data), 10):
        for j in range(10):
            axes[i, j].axis('off')

    # save the figure
    plt.tight_layout()
    plt.savefig('SHAP/shap_values.png')
    plt.close()


def main():
    model = buildModel()
    model.load_state_dict(torch.load('mnist_cnn.pt'))

    if dataset == "MNIST":
        train_loader, test_loader = load_MNIST_data()
    if dataset == "CIFAR10":
        train_loader, test_loader = load_CIFAR10_data()

    # Define the SHAP explainer
    save_and_plot_shap_values(test_loader, model)

if __name__ == '__main__':
    main()

