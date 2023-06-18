import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap
from util import load_MNIST_data, build_MNIST_Model, clear_memory

def getData(dataloader):
    """ 
    Get one sample of each class from the dataloader.
    """
    data = []
    target = []

    for batch_idx, (data_batch, target_batch) in enumerate(dataloader):
        for data_batch_i, target_batch_i in zip(data_batch, target_batch):
            if target_batch_i not in target:
                target.append(target_batch_i)
                data.append(data_batch_i)

    # Sort the data and target based on the target
    data, target = zip(*sorted(zip(data, target), key=lambda x: x[1]))

    data = torch.stack(data, dim=0)
    target = torch.tensor(target)

    return data, target

def save_and_plot_shap_values(dataloader, model):
    """
    Generate and save SHAP values for the given model and dataloader.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clear_memory()

    data, target = getData(dataloader)
    data = data.to(device)
    target = target.to(device)

    # Move model to device
    model = model.to(device)

    # Generate SHAP values
    explainer = shap.DeepExplainer(model, data)
    shap_values = explainer.shap_values(data)

    save_dir = 'SHAP'
    os.makedirs(save_dir, exist_ok=True)

    # Create a 10x10 grid of subplots
    fig, axes = plt.subplots(10, 11, figsize=(20, 22))

    # Iterate over the SHAP values and plot on the subplots
    for i in range(len(data)):
        shap_i = shap_values[i]
        label = target[i].item()

        # Save the original image as a numpy array
        np.save(f'{save_dir}/{i}_original.npy', data[i].cpu().numpy())

        # Plot the original image
        axes[i, 0].imshow(data[i].cpu().reshape(28, 28), cmap='gray')
        axes[i, 0].set_title(f'Label: {label}')

        # Plot the SHAP values
        num_shap_values = min(10, len(shap_i))  # Adjust the number of SHAP values to fit within the grid
        for j in range(num_shap_values):
            # Save the SHAP value as a numpy array
            np.save(f'{save_dir}/{i}_shap_{j}.npy', shap_i[j])
            img = axes[i, j+1].imshow(shap_i[j].reshape(28, 28), cmap='jet')
            axes[i, j+1].axis('off')
            axes[i, j+1].set_title(f'SHAP value {j+1}')

        # Remove empty cells in the row
        for j in range(num_shap_values + 1, 11):
            axes[i, j].axis('off')

        # Save the row individually and remove the white space
        row_fig = plt.figure(figsize=(10, 1))
        row_axes = row_fig.subplots(1, num_shap_values + 1)
        row_axes[0].imshow(data[i].cpu().reshape(28, 28), cmap='gray')
        row_axes[0].set_title(f'Label: {label}')
        for j in range(num_shap_values):
            row_axes[j+1].imshow(shap_i[j].reshape(28, 28), cmap='jet')
            row_axes[j+1].axis('off')
        plt.tight_layout()
        row_fig.savefig(f'{save_dir}/row_{i}.png')
        plt.close(row_fig)

    # Remove empty rows
    for i in range(len(data), 10):
        for j in range(10):
            axes[i, j].axis('off')

    # Add colorbar
    cbar_ax = fig.add_axes([.93, 0.15, 0.02, 0.7])  # Adjust the position of the colorbar
    fig.colorbar(img, cax=cbar_ax)

    # Save the figure
    # plt.tight_layout()
    plt.savefig(f'{save_dir}/shap_values.png')
    plt.close()

def main():
    train_loader, test_loader = load_MNIST_data()
    model = build_MNIST_Model()
    model.load_state_dict(torch.load('MNIST_cnn.pt'))

    # Define the SHAP explainer
    save_and_plot_shap_values(test_loader, model)

if __name__ == '__main__':
    main()
