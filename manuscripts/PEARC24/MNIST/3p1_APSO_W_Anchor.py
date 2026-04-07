# --- Imports ---
import os
import torch
import umap
import numpy as np
import tqdm
import matplotlib.pyplot as plt

import Adversarial_Observation as AO
from Adversarial_Observation.Swarm import ParticleSwarm  

# --- Global Config ---
optimize = 3  # Target class for the attack


# --- Main Function ---
def main():
    # Load model and data
    model = AO.utils.load_MNIST_model()
    train_loader, test_loader = AO.utils.load_MNIST_data()

    if os.path.isfile('MNIST_cnn.pt'):
        model.load_state_dict(torch.load('MNIST_cnn.pt'))
    else:
        raise FileNotFoundError("MNIST_cnn.pt not found. Please run 1_build_train.py first.")

    model.eval()

    # Setup UMAP and collect test points
    umap_model = umap.UMAP()
    accumulated_data = []
    targets = []
    otherpoints = {}
    anchor = None

    for idx, (data, target) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader), desc='Preparing UMAP'):
        if idx > 10:
            break
        for img, label in zip(data, target):
            if label.item() == optimize and anchor is None:
                anchor = img
            img_np = img.reshape(1, 28 * 28).detach().cpu().numpy()
            accumulated_data.append(img_np)
            targets.append(label)

    accumulated_data = np.concatenate(accumulated_data, axis=0)
    reduced = umap_model.fit_transform(accumulated_data)

    for idx, label in tqdm.tqdm(enumerate(targets), total=len(targets), desc='Storing UMAP data'):
        label_value = label.item()
        if label_value not in otherpoints:
            otherpoints[label_value] = []
        otherpoints[label_value].append(reduced[idx])

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Swarm parameters
    num_particles = 300
    shape = 28 * 28
    epochs = 20

    initial_points = np.random.rand(num_particles, shape)
    mask = np.random.choice([0, 1], size=initial_points.shape, p=[0.99, 0.01])
    initial_points *= mask
    initial_points = torch.tensor(initial_points, dtype=torch.float32)

    # Add anchor to the swarm
    if anchor is None:
        raise ValueError(f"No anchor found for class {optimize}.")
    initial_points = torch.cat((initial_points, anchor.reshape(1, shape)), dim=0)

    # Run the particle swarm
    runSwarm(initial_points, model, device, umap_model, epochs, otherpoints)


# --- Run Swarm with Custom ParticleSwarm ---
def runSwarm(initial_points, model, device, umap_model, epochs, otherpoints):
    swarm = ParticleSwarm(
        model=model,
        input_set=initial_points,
        starting_class=optimize,  # assuming starting and target are same
        target_class=optimize,
        num_iterations=epochs,
        save_dir='./APSO_A',
        enable_logging=False,
        device=str(device)
    )

    plotSwarm(swarm, umap_model, 0, otherpoints)
    swarm.optimize()

    # After optimization, save final results
    plotSwarm(swarm, umap_model, epochs, otherpoints)


# --- Plotting Functions ---
def plotSwarm(swarm, umap_model, epoch, otherpoints):
    fig, ax = plt.subplots(figsize=(12, 12))

    # Plot static test points
    for key in otherpoints:
        x = [i[0] for i in otherpoints[key]]
        y = [i[1] for i in otherpoints[key]]
        ax.scatter(x, y, label=key)

    # Swarm points
    points = np.array(swarm.getPoints())
    if points.ndim > 2:
        points = points.reshape(points.shape[0], -1)

    transformed = umap_model.transform(points)
    ax.scatter(transformed[:, 0], transformed[:, 1], c='black', label='Swarm')

    ax.legend()
    ax.set_title(f'Epoch: {epoch}')
    os.makedirs('./APSO_A/points', exist_ok=True)
    plt.savefig(f'./APSO_A/points/epoch_{epoch}.png')
    plt.close()

    plotImages(swarm, epoch)


def plotImages(swarm, epoch):
    points = np.array(swarm.getPoints())
    points = points.reshape(-1, 1, 28, 28)

    best = swarm.getBest().reshape(28, 28)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Plot best particle
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(best, cmap='gray')
    ax.axis('off')
    confidence = swarm.model(torch.tensor(best).reshape(1, 1, 28, 28).float().to(device))[0][optimize].item()
    ax.set_title(f'Confidence of {optimize}: {confidence:.4f}')

    grad = AO.Attacks.gradient_map(
        torch.tensor(best).reshape(1, 1, 28, 28).float().to(device), swarm.model,
        (1, 1, 28, 28)
    )[0].reshape(28, 28)

    grad = np.abs(grad)
    grad_norm = (grad - np.min(grad)) / (np.max(grad) - np.min(grad) + 1e-8)
    ax.imshow(plt.get_cmap('jet')(grad_norm), alpha=0.7)

    os.makedirs(f'./APSO_A/images/epoch_{epoch}', exist_ok=True)
    plt.savefig(f'./APSO_A/images/epoch_{epoch}/best.png')
    plt.close()

    # Plot remaining particles
    for idx, point in enumerate(points):
        point_img = point.reshape(28, 28)
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(point_img, cmap='gray')
        ax.axis('off')
        confidence = swarm.model(torch.tensor(point_img).reshape(1, 1, 28, 28).float().to(device))[0][optimize].item()
        ax.set_title(f'Confidence of {optimize}: {confidence:.4f}')

        grad = AO.Attacks.gradient_map(
            torch.tensor(point_img).reshape(1, 1, 28, 28).float().to(device),
            swarm.model,
            (1, 1, 28, 28)
        )[0].reshape(28, 28)
        grad = np.abs(grad)
        grad_norm = (grad - np.min(grad)) / (np.max(grad) - np.min(grad) + 1e-8)
        ax.imshow(plt.get_cmap('jet')(grad_norm), alpha=0.7)

        plt.tight_layout()
        plt.savefig(f'./APSO_A/images/epoch_{epoch}/point_{idx}.png')
        plt.close()


# --- Entry ---
if __name__ == '__main__':
    main()
