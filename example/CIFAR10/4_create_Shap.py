import torch
import torchvision
import shap
import numpy as np
import matplotlib.pyplot as plt
from util import *
import os
from util import *

def getData(dataloader):
    """ 
    gets one sample of each class from the dataloader
    """
    data = []
    target = []

    for batch_idx, (data_batch, target_batch) in enumerate(dataloader):
        for data_batch_i, target_batch_i in zip(data_batch,target_batch):
            if target_batch_i not in target:
                target.append(target_batch_i)
                data.append(data_batch_i)

    # sort the data and target based on the target
    data, target = zip(*sorted(zip(data, target), key=lambda x: x[1]))

    data = torch.stack(data,dim=0)
    target = torch.tensor(target)

    return data, target

def save_and_plot_shap_values(dataloader, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clear_memory()

    data, target = getData(dataloader)
    data = data.to(device)
    target = target.to(device)

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
        shap_i = np.array(shap_i)
        shap_i = shap_i.transpose(0,2, 3, 1)
        label = target[i]


        # plot the original image
        data_i = np.array(data[i].cpu()).transpose(1,2,0)
        
        #  convert tensor to numpy
        axes[i, 0].imshow(data_i)
        axes[i, 0].set_title(f'Label: {label}')

        # plot the SHAP values
        num_shap_values = min(10, len(shap_i))  # Adjust the number of SHAP values to fit within the grid
        for j in range(num_shap_values):

            axes[i, j+1].imshow(shap_i[j]/shap_i[j].max(), cmap='jet')
            axes[i, j+1].axis('off')
            axes[i, j+1].set_title(f'SHAP value: {j}')

        # remove empty cells in the row
        for j in range(num_shap_values + 1, 11):
            axes[i, j].axis('off')

        # save the row individually
        row_fig = plt.figure(figsize=(8, 1))
        row_axes = row_fig.subplots(1, num_shap_values + 1)
        
        row_axes[0].imshow(data[i].cpu().numpy().transpose(1,2,0))
        row_axes[0].set_title(f'Label: {label}')
        for j in range(num_shap_values):
            row_axes[j+1].imshow(-shap_i[j]/shap_i[j].max(), cmap='jet')
            row_axes[j+1].axis('off')
            # row_axes[j+1].set_title(f'SHAP value: {j}')
        plt.tight_layout()
        row_fig.savefig(f'SHAP/row_{i}.png')
        plt.close(row_fig)
        
    # remove empty rows
    for i in range(len(data), 10):
        for j in range(10):
            axes[i, j].axis('off')

    # save the figure
    plt.tight_layout()
    plt.savefig('SHAP/shap_values.png')
    plt.close()


def main():
    train_loader, test_loader = load_CIFAR10_data()
    model = build_CIFAR10_Model()
    model.load_state_dict(torch.load('CIFAR10_cnn.pt'))

    # Define the SHAP explainer
    save_and_plot_shap_values(test_loader, model)

if __name__ == '__main__':
    main()