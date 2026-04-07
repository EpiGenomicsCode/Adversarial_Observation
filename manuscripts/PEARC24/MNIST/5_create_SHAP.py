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
    shap_values = explainer.shap_values(data)  
    
    # --- ROBUST SHAP SHAPE NORMALIZATION ---
    # SHAP can return a list of 10 arrays OR a list of 1 array containing all classes.
    # This block forces the data into a standard shape: (batch_size, num_classes, 28, 28)
    if isinstance(shap_values, list):
        if len(shap_values) == 10:
            # Case A: List of 10 classes. Convert to array and swap axes to (batch, class, ...)
            shap_tensor = np.array(shap_values).swapaxes(0, 1)
        elif len(shap_values) == 1:
            # Case B: List of 1 containing everything. Extract the array directly.
            shap_tensor = np.array(shap_values[0])
        else:
            shap_tensor = np.array(shap_values)
    else:
        # Case C: Returned a raw numpy array right out of the gate
        shap_tensor = np.array(shap_values)
        
    # Flatten out the channel dimension and strictly enforce (10_images, 10_classes, 28, 28)
    shap_tensor = shap_tensor.reshape(len(data), 10, 28, 28)
    # ---------------------------------------

    save_dir = 'SHAP'
    os.makedirs(save_dir, exist_ok=True)

    # Create a 10x11 grid: 1 original + 10 SHAP values
    fig, axes = plt.subplots(10, 11, figsize=(20, 22))
    last_img = None  

    for i in range(len(data)):
        label = target[i].item()

        # Save original image
        np.save(f'{save_dir}/{i}_original.npy', data[i].cpu().numpy())
        axes[i, 0].imshow(data[i].cpu().reshape(28, 28), cmap='gray')
        axes[i, 0].set_title(f'Label: {label}')
        axes[i, 0].axis('off')

        # 1. Main Grid Plotting
        for j in range(10):
            reshaped = shap_tensor[i, j] # Safely extracts the exact 28x28 grid
            np.save(f'{save_dir}/{i}_shap_{j}.npy', reshaped)
            last_img = axes[i, j+1].imshow(reshaped, cmap='jet')
            axes[i, j+1].axis('off')

        # 2. Save row as standalone image
        row_fig, row_axes = plt.subplots(1, 11, figsize=(20, 2))
        row_axes[0].imshow(data[i].cpu().reshape(28, 28), cmap='gray')
        row_axes[0].set_title(f'Label: {label}')
        row_axes[0].axis('off')
        
        for j in range(10):
            row_axes[j+1].imshow(shap_tensor[i, j], cmap='jet')
            row_axes[j+1].axis('off')
            
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
