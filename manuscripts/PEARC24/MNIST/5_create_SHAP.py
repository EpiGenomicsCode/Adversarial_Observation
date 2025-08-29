import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import shap
import  Adversarial_Observation as AO
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

    data, target = getData(dataloader)
    data = data.to(device)
    target = target.to(device)

    model = model.to(device)

    explainer = shap.DeepExplainer(model, data)
    shap_values = explainer.shap_values(data)  # List of [class][samples, features]

    save_dir = 'SHAP'
    os.makedirs(save_dir, exist_ok=True)

    # Create a 10x11 grid: 1 original + 10 SHAP values
    fig, axes = plt.subplots(10, 11, figsize=(20, 22))
    last_img = None  # For colorbar

    for i in range(len(data)):
        label = target[i].item()
        shap_i = [class_shap[i] for class_shap in shap_values]  # SHAP per class, for this image

        # Save original image
        np.save(f'{save_dir}/{i}_original.npy', data[i].cpu().numpy())
        axes[i, 0].imshow(data[i].cpu().reshape(28, 28), cmap='gray')
        axes[i, 0].set_title(f'Label: {label}')
        axes[i, 0].axis('off')

        for j in range(min(10, len(shap_i))):
            shap_array = shap_i[j]
            try:
                reshaped = shap_array.reshape(10, 28, 28)[j]  # extract correct class
            except Exception as e:
                print(f"[ERROR] SHAP reshape failed for sample {i}, class {j}: {e}")
                continue

            np.save(f'{save_dir}/{i}_shap_{j}.npy', shap_array)
            last_img = axes[i, j+1].imshow(reshaped, cmap='jet')
            axes[i, j+1].axis('off')


        # Fill remaining columns
        for j in range(len(shap_i) + 1, 11):
            axes[i, j].axis('off')

        # Save row as standalone image
        row_fig, row_axes = plt.subplots(1, 11, figsize=(20, 2))
        row_axes[0].imshow(data[i].cpu().reshape(28, 28), cmap='gray')
        row_axes[0].set_title(f'Label: {label}')
        row_axes[0].axis('off')
        for j in range(min(10, len(shap_i))):
            row_axes[j+1].imshow(shap_i[j][:784].reshape(28, 28), cmap='jet')
            row_axes[j+1].axis('off')
        for j in range(len(shap_i) + 1, 11):
            row_axes[j].axis('off')
        plt.tight_layout()
        row_fig.savefig(f'{save_dir}/row_{i}.png')
        plt.close(row_fig)

    # Fill empty rows if less than 10 samples
    for i in range(len(data), 10):
        for j in range(11):
            axes[i, j].axis('off')

    # Add colorbar only if a SHAP plot was rendered
    if last_img is not None:
        cbar_ax = fig.add_axes([.93, 0.15, 0.02, 0.7])
        fig.colorbar(last_img, cax=cbar_ax)

    plt.tight_layout()
    plt.savefig(f'{save_dir}/shap_values.png')
    plt.close()

def main():
    train_loader, test_loader = AO.utils.load_MNIST_data()
    model = AO.load_MNIST_model()
    model.load_state_dict(torch.load('MNIST_cnn.pt'))

    # Define the SHAP explainer
    save_and_plot_shap_values(test_loader, model)

if __name__ == '__main__':
    main()
