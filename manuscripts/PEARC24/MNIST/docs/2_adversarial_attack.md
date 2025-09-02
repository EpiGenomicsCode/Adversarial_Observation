## Adversarial Image Generation and Evaluation

This project demonstrates the generation of adversarial images using the **Fast Gradient Sign Method (FGSM)** and evaluates their impact on a pre-trained CNN model trained on the **MNIST dataset**. The script, `2_adversarial_attack.py`, generates perturbed images and visualizes the differences between the original and adversarial images.

### Functions

The code includes several functions that facilitate adversarial image generation, evaluation, and visualization.

#### 1. `seedEverything(seed)`

Seeds all random number generators to ensure reproducibility of results.

* **Parameters**:

  * `seed`: An integer seed value for the random number generators.

#### 2. `main()`

The main function that orchestrates the adversarial image generation process. It performs the following tasks:

* Seeds the random number generators for reproducibility.
* Loads the MNIST dataset and pre-trained model.
* Sets the model to evaluation mode.
* Generates adversarial images using the FGSM attack with varying epsilon values.
* Saves the adversarial images, the perturbations, activation maps, and original images for analysis.

#### 3. `grad_ascent(model, device)`

This function performs **gradient ascent** to create adversarial examples for each class by perturbing an image to maximize the model’s confidence for the target class. It saves generated images as PNG files in the `./gradient_Ascent` directory.

* **Parameters**:

  * `model`: The pre-trained CNN model.
  * `device`: The computing device (CPU or GPU).

#### 4. `fgsm(imgs, labels, model, device)`

Generates adversarial images using the **Fast Gradient Sign Method (FGSM)** with a list of epsilon values. The adversarial images, their gradients, and the activation maps are saved to the `./attack_results` directory.

* **Parameters**:

  * `imgs`: The input images for generating adversarial examples.
  * `labels`: The true labels of the input images.
  * `model`: The pre-trained CNN model.
  * `device`: The computing device (CPU or GPU).

#### 5. `plot_perterbed(perterbed, eps, imgs, labels, model)`

Visualizes the adversarial images, their corresponding perturbations, and activation maps. The original, perturbed, and noise images are saved as PNG files, and their numpy arrays are stored in the `./attack_results` directory.

* **Parameters**:

  * `perterbed`: The perturbed images.
  * `eps`: The epsilon value used in the FGSM attack.
  * `imgs`: The original input images.
  * `labels`: The true labels of the input images.
  * `model`: The pre-trained CNN model.

### Usage

To use this code, follow these steps:

1. **Install Dependencies**: Ensure that all the necessary dependencies (PyTorch, torchvision, numpy, matplotlib, etc.) are installed.

2. **Run the Script**: Execute the Python script `2_adversarial_attack.py` to generate adversarial images and evaluate their impact.

   ```bash
   python 2_adversarial_attack.py
   ```

3. **Generated Results**:

   * The generated adversarial images and the corresponding perturbations will be saved in the `./attack_results` directory.
   * The visualizations of the original, perturbed, and noise images, along with the activation maps, will be saved as PNG files.
   * Numpy arrays of the perturbed images are also saved in the same directory for further analysis.

4. **Adjusting Attack Parameters**:

   * The `eps` values (perturbation strength) used in the FGSM attack can be adjusted in the code.
   * You can modify the saving directories and file names as needed.

### Prerequisites

* **Pre-trained Model**: Ensure that the pre-trained model weights `MNIST_cnn.pt` are available in the working directory. If not, you need to run the model training script (e.g., `1_build_train.py`) to obtain the `MNIST_cnn.pt` file.

* **Paths**: Make sure the file paths for saving images and results match your desired directory structure.

### Notes

* This code supports both **CPU** and **GPU** execution. The device is automatically detected.
* **Gradient ascent** and **FGSM** can be used to create adversarial examples, with results saved for each class and epsilon value.
* The perturbation strength (`eps`) can be customized to explore the effect of different attack intensities.

### Output Structure

* `./attack_results/`: Directory containing:

  * `fgsm_{eps}_{label}.png`: Original, perturbed, and noise images for each attack.
  * `fgsm_{eps}_{label}.npy`: Numpy array of the perturbed image.
  * `fgsm_{eps}_{label}_grad.png`: Activation maps of original and perturbed images.
* `./gradient_Ascent/`: Directory containing:

  * `ga_{i}.png`: Gradient ascent images for each class.
