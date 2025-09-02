# Adversarial Particle Swarm Optimization (APSO) Description

The provided Python script implements the **Adversarial Particle Swarm Optimization (APSO)** algorithm to generate adversarial examples for a machine learning model. APSO optimizes a cost function using Particle Swarm Optimization (PSO) to perturb input data (MNIST images) in a way that maximizes the model's misclassification probability. The generated adversarial examples are visualized using UMAP for dimensionality reduction and gradient-based techniques to highlight regions of vulnerability.

## Global Variables

* **`label`**: The target label for the adversarial attack (set to `3` by default). This is the class the algorithm tries to generate adversarial examples for.
* **`initial`**: The initial label of the dataset (not used directly in the code but could be a placeholder for tracking the original label).
* **`epochs`**: The number of iterations the PSO algorithm will run to optimize the adversarial examples (set to `20`).
* **`points`**: The number of particles in the swarm (set to `50`).

## Functions

### `cost_func(model, point)`

This function calculates the cost (prediction score) for a given input data point using the pre-trained PyTorch model. It computes the product of the model’s confidence in predicting the target `label` and the sum of the gradients over the image's pixels. The function returns the cost, which represents how well the current point (image) serves as an adversarial example.

**Parameters:**

* `model`: The pre-trained PyTorch model used to classify the data.
* `point`: The input data point (image) to evaluate.

**Returns:**

* A float value representing the adversarial cost.

### `plotSwarm(swarm, umap_model, epoch, otherpoints)`

This function visualizes the state of the particle swarm during optimization. It uses UMAP for dimensionality reduction and generates a scatter plot of the swarm's position in 2D space, along with the positions of the original class data points (MNIST images).

**Parameters:**

* `swarm`: The current state of the swarm (ParticleSwarm object).
* `umap_model`: The trained UMAP model to reduce the dimensionality of data for plotting.
* `epoch`: The current epoch number to label the plot.
* `otherpoints`: A dictionary mapping the target labels to their corresponding UMAP-reduced data points.

### `plotImages(swarm, epoch)`

This function visualizes the generated adversarial images and their corresponding gradient maps (showing the influence of each pixel on the model’s classification). It saves the visualizations of the best particles and others in the swarm at each epoch.

**Parameters:**

* `swarm`: The current state of the swarm (ParticleSwarm object).
* `epoch`: The current epoch number.

### `runSwarm(initial_points, model, device, umap_model, epochs, otherpoints)`

This function initializes and runs the particle swarm optimization process. It uses the `Swarm.ParticleSwarm` class to optimize a set of particles (initial adversarial candidates) for a specified number of epochs. After optimization, the best adversarial examples are returned.

**Parameters:**

* `initial_points`: The initial swarm points (random adversarial candidates).
* `model`: The pre-trained PyTorch model used to classify the data.
* `device`: The device (`'cuda'` or `'cpu'`) on which to run the model.
* `umap_model`: The UMAP model to reduce the dimensionality for visualization.
* `epochs`: The number of iterations for optimization.
* `otherpoints`: A dictionary mapping target labels to their corresponding data points.

### `main()`

This function is the main entry point of the script. It initializes global variables, loads the dataset (MNIST), builds the model, and runs APSO for generating adversarial examples. It also handles saving and visualizing the results during the optimization process.

**Execution flow:**

1. Load the MNIST dataset and pre-trained model.
2. Initialize UMAP and other required variables.
3. Run the `runSwarm` function to perform optimization.
4. Plot and save the results (including UMAP visualizations and adversarial images).

## Execution

### Prerequisites

To run this script, you need:

* **PyTorch**: For building and running the neural network model.
* **UMAP**: For dimensionality reduction and visualization.
* **tqdm**: For progress bars.
* **matplotlib**: For plotting the swarm and images.
* **SciPy**: For sparse matrix operations (if needed).

### Running the Script

To execute the script, follow these steps:

1. **Install required dependencies**:

   ```bash
   pip install torch umap-learn matplotlib scipy tqdm
   ```

2. **Download or train the MNIST model**:

   * If you have the pre-trained `MNIST_cnn.pt` model, place it in the working directory.
   * If not, you need to train a model first using the appropriate training script (e.g., `1_build_train.py`).

3. **Run the script**:

   ```bash
   python adversarial_pso.py
   ```

4. **Output**:

   * The script will generate plots of the adversarial examples at each epoch and save them to the `./APSO/images/` directory.
   * The swarm optimization results (including UMAP visualizations and gradient maps) will be saved under `./APSO/points/` and `./APSO/images/`.

### Visualization

The generated images and plots will allow you to observe how the particles (adversarial examples) evolve over time to deceive the model into misclassifying them as the target class.

## Notes

* **Swarm Configuration**: The PSO parameters like inertia weight, cognitive weight, and social weight are set to standard values but can be tuned for different optimization behavior.
* **Visualization**: The code generates and saves multiple visualizations, including UMAP scatter plots and adversarial images with their corresponding gradient maps. These help to understand how the particles are evolving and which parts of the image influence the model's decision.
* **File Storage**: The script saves the results in `./APSO/images/` and `./APSO/points/`. Ensure enough disk space for storing images.

## References

* [PyTorch](https://pytorch.org/): Deep learning framework used for the MNIST model.
* [UMAP](https://umap-learn.readthedocs.io/): Used for dimensionality reduction and visualizing high-dimensional data.
* [Particle Swarm Optimization (PSO)](https://en.wikipedia.org/wiki/Particle_swarm_optimization): Optimization algorithm inspired by the social behavior of birds flocking or fish schooling.
