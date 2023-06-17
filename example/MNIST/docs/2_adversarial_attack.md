## Adversarial Image Generation and Evaluation

This code, implemented in `2_adversarial_attack.py`, demonstrates the generation of adversarial images using the Fast Gradient Sign Method (FGSM) and evaluates their impact on the model's predictions. The code is designed to work with a pre-trained CNN model on the MNIST dataset.


### Functions

The code includes the following functions:

#### 1. `seedEverything(seed)`

This function seeds all the random number generators to ensure reproducibility.

- `seed`: The seed value for random number generators.

#### 2. `main()`

This is the main function that orchestrates the adversarial image generation and evaluation process. It performs the following steps:

- Seeds the random number generators for reproducibility.
- Loads the MNIST dataset using the `load_MNIST_data()` function from the `util` module.
- Builds the MNIST CNN model using the `build_MNIST_Model()` function from the `util` module.
- Loads the pre-trained model weights from the `MNIST_cnn.pt` file.
- Sets the model to evaluation mode.
- Generates adversarial images using the FGSM attack on misclassified images.
- Saves the generated adversarial images, their corresponding perturbations, activation maps, and original images for reference.

### Usage

To use this code:

1. Make sure you have the necessary dependencies installed.
2. Run the Python script named `2_adversarial_attack.py`.
3. Execute the script to generate adversarial images using the FGSM attack and evaluate their impact.
4. The generated adversarial images will be saved in the `Noise/fgsm` directory.
5. The corresponding perturbations, activation maps, and original images will also be saved for each adversarial image.
6. Adjust the attack parameters (`eps` values) and saving directories as needed.

Note: Before running the code, make sure to have the pre-trained model weights in the `MNIST_cnn.pt` file and modify the paths and filenames to match your desired configuration.
