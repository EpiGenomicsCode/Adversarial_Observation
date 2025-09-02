
# Adversarial Particle Swarm Optimization (APSO) with Custom Swarm Class

This Python script implements an **Adversarial Particle Swarm Optimization (APSO)** algorithm to generate adversarial examples for a **MNIST** model. The algorithm uses **Particle Swarm Optimization (PSO)** to perturb input images in a way that maximizes the misclassification probability. The generated adversarial examples are visualized using **UMAP** for dimensionality reduction and gradient-based techniques to highlight areas of high vulnerability in the model.

The script uses a custom `ParticleSwarm` class defined in the `Adversarial_Observation` package to perform the optimization.

## Global Variables

* **`optimize`**: The target class label for the adversarial attack (default is `3`).
* **`epochs`**: The number of optimization iterations for the swarm (set to `20`).
* **`num_particles`**: The number of particles in the swarm (set to `300`).
* **`shape`**: The dimensionality of each particle (flattened MNIST images, i.e., `28 * 28 = 784`).

## Functions

### `main()`

The main entry point of the script. This function:

1. Loads the pre-trained MNIST model.
2. Loads the MNIST data using PyTorch's DataLoader.
3. Initializes **UMAP** for dimensionality reduction and prepares test points for visualization.
4. Sets up the particle swarm by initializing random particles and adding an anchor point for the target class.
5. Runs the **particle swarm optimization** by calling the `runSwarm()` function.

### `runSwarm(initial_points, model, device, umap_model, epochs, otherpoints)`

This function runs the **Particle Swarm Optimization (PSO)** algorithm. It initializes a **ParticleSwarm** object and optimizes the particles to generate adversarial examples that target the specified class (`optimize`). The function:

1. Initializes the swarm with random starting points (particles).
2. Calls the `optimize()` method of the **ParticleSwarm** class to perform the optimization.
3. Saves and plots the swarm’s progress during each epoch.

### `plotSwarm(swarm, umap_model, epoch, otherpoints)`

This function visualizes the positions of the particles in the swarm at each optimization epoch. It:

1. Visualizes the particles using **UMAP** to reduce the dimensionality of the swarm's position in 2D.
2. Saves the scatter plot as an image to the `./APSO_A/points/` directory.
3. Calls `plotImages()` to visualize the adversarial images.

### `plotImages(swarm, epoch)`

This function visualizes the adversarial images and their gradient maps (indicating which parts of the image most influence the model’s decision). It:

1. Plots the **best particle** (adversarial example) and its confidence score for the target class.
2. Visualizes the **gradient map** for the best particle, showing which pixels influence the model's decision the most.
3. Saves the images of the best particle and all particles in the swarm for each epoch to the `./APSO_A/images/` directory.

## Execution

### Prerequisites

To run this script, you will need the following dependencies:

* **PyTorch**: For model training and inference.
* **UMAP**: For dimensionality reduction and visualization.
* **Matplotlib**: For plotting images and swarm positions.
* **tqdm**: For progress bars during the execution.
* **Adversarial\_Observation**: This is a custom package that contains the `ParticleSwarm` class and related functionality for adversarial attacks.

Install the necessary dependencies:

```bash
pip install torch umap-learn matplotlib tqdm
```

Additionally, the **Adversarial\_Observation** package must be available. If you don't have this package, ensure that it's installed or accessible in your project directory.

### Model Preparation

The script assumes that a pre-trained MNIST model (`MNIST_cnn.pt`) is available. If you don’t have this model file, you need to train it using the provided training script (`1_build_train.py`) before running the adversarial attack script.

### Running the Script

To execute the script:

1. Ensure you have the pre-trained model `MNIST_cnn.pt` in the current working directory.
2. Run the script as follows:

   ```bash
   python adversarial_pso.py
   ```

### Output

* The generated adversarial examples and swarm positions will be saved to the `./APSO_A/points/` and `./APSO_A/images/` directories.
* The script saves **UMAP visualizations** and **gradient maps** of the best particle at each optimization epoch.

### Visualization

The **swarm visualizations** show how the adversarial particles evolve during the optimization process. The **image visualizations** show the generated adversarial examples and highlight the pixels that most influence the model's classification using a gradient map.

## Notes

* **Swarm Configuration**: The swarm's configuration, including the number of particles, epochs, and particle initialization, can be adjusted by modifying the relevant parameters in the script.
* **Anchor**: The algorithm selects one test image with the target label (`optimize`) as the anchor point for the attack. If no anchor image is found in the dataset, a `ValueError` will be raised.
* **Visualization**: The code generates and saves multiple visualizations, including UMAP scatter plots, adversarial images, and their gradient maps. These help to understand the impact of the optimization process on the model.

## References

* **PyTorch**: [https://pytorch.org/](https://pytorch.org/) – Deep learning framework used to build and run the MNIST model.
* **UMAP**: [https://umap-learn.readthedocs.io/](https://umap-learn.readthedocs.io/) – Dimensionality reduction technique used for visualizing high-dimensional data.
* **Particle Swarm Optimization (PSO)**: [https://en.wikipedia.org/wiki/Particle\_swarm\_optimization](https://en.wikipedia.org/wiki/Particle_swarm_optimization) – Optimization algorithm used for generating adversarial examples.

