# Adversarial Attack Workflow with Particle Swarm Optimization

This repository contains a framework for generating **adversarial attacks** on a pre-trained or newly trained on image classification models using **Particle Swarm Optimization (PSO)**. The workflow includes model training, adversarial attack generation, explainability visualizations, and detailed analysis of attack results.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Setup and Installation](#setup-and-installation)
4. [Package Structure](#package-structure)
5. [Usage](#usage)
   * [Train a New Model](#train-a-new-model)
   * [Load a Pre-trained Model](#load-a-pre-trained-model)
   * [Perform Adversarial Attack](#perform-adversarial-attack)
6. [Directory Structure](#directory-structure)
7. [Results and Analysis](#results-and-analysis)
8. [Documentation](#documentation)
9. [Contributing](#contributing)
10. [Citing This Work](#citing-this-work)
11. [License](#license)

---

## Overview

This project demonstrates how to attack a **PyTorch-based MNIST classifier** using several **adversarial attack** methods and explainability techniques. The main capabilities include:

* **Model Training:** Create and train a convolutional neural network (CNN) for classification using PyTorch.
* **Adversarial Attacks:** Black-box adversarial attack that uses swarm intelligence to generate perturbations causing misclassification.
* **Analysis:** Collect detailed metrics during attacks, including confidence values, softmax outputs, and pixel-wise differences from the original image.

The model can either be trained from scratch or loaded from a pre-trained checkpoint. Attack results are saved with detailed logs and images for further analysis.

---

## Requirements

This project requires the following Python libraries:

* `torch` / `torchvision` (for model building, training, and data loading)
* `numpy` (for numerical operations)
* `matplotlib` (for visualizations)
* `tqdm` (for progress bars)
* `scipy` (for utility functions)

You can install the necessary dependencies by running:

```bash
pip install -r requirements.txt
```

---

## Setup and Installation

1. **Clone the repository:**

```bash
git clone https://github.com/EpiGenomicsCode/Adversarial_Observation.git
cd Adversarial_Observation
```

2. **Install dependencies:**

```bash
pip install -r requirements.txt
```

3. **Install the package** (optional, for importable module usage):

```bash
pip install -e .
```

4. **Run the script** with the desired parameters.

---

## Package Structure

The `Adversarial_Observation` package is organized into the following modules:

* **`Attacks`** — Core attack and explainability methods:
  * `fgsm_attack()` — FGSM adversarial attack
  * `gradient_ascent()` — Neuron activation maximization via gradient ascent
  * `gradient_map()` — Gradient-based input attribution (vanilla, guided, ReLU backprop)
  * `saliency_map()` — Saliency map generation for a target class
* **`utils`** — Data loading, model loading, metrics, and reproducibility:
  * `load_MNIST_data()` — Load MNIST train/test data loaders
  * `load_MNIST_model()` — Load a sequential CNN model
  * `fgsm_attack()` — Standalone FGSM utility with device support
  * `compute_success_rate()` — Compute attack success rate
  * `log_metrics()` — Log success rate and perturbation magnitude
  * `seed_everything()` — Set random seeds for reproducibility
  * `visualize_adversarial_examples()` — Plot original vs. adversarial images
* **`visualize`** — Animation and visualization:
  * `visualize_gif()` — Generate GIF animations of attack progression

---

## Usage

### Train a New Model

To train a new **MNIST classifier** model from scratch, run:

```bash
python taint_MNIST.py --iterations 50 --particles 100 --save_dir "analysis_results"
```

This command will train the model on the MNIST dataset and save the trained model if no pre-trained model path is provided.

### Load a Pre-trained Model

If you already have a pre-trained model, load it with the `--model_path` argument:

```bash
python taint_MNIST.py --model_path "path_to_model/mnist_model.pt" --iterations 50 --particles 100 --save_dir "analysis_results"
```

This will load the provided pre-trained model, evaluate it on the test dataset, and then perform the adversarial attack.

### Perform Adversarial Attack

Once the model is trained or loaded, the script will automatically perform a **black-box adversarial attack** on a specified image in the test dataset using **Particle Swarm Optimization (PSO)**.

**Example:**

```bash
python taint_MNIST.py --iterations 50 --particles 100 --save_dir "analysis_results"
```

This command performs the PSO attack with **50 iterations** and **100 particles**.

---

## Directory Structure

After running the attack, the results will be saved in the `analysis_results` directory (or the directory specified by `--save_dir`):

```
analysis_results/
│
├── original.png                  # Original image before attack
├── iteration_1/                  # Directory for each iteration
│   ├── attack-vector_image_1.png  # Perturbed image for particle 1 at iteration 1
│   ├── attack-vector_image_2.png  # Perturbed image for particle 2 at iteration 1
│   └── ...
├── iteration_2/
│   ├── attack-vector_image_1.png
│   └── ...
├── attack_analysis.json           # JSON file containing analysis results
└── ...
```

### Key Files

* **`original.png`**: The original image before the attack.
* **`attack-vector_image_*.png`**: Perturbed images generated by particles at each iteration.
* **`attack_analysis.json`**: Analysis of the attack including confidence values, perturbation differences, and more.

---

## Results and Analysis

After the attack is complete, the following information is saved:

* **Images** showing pixel-wise differences between the original image and perturbed versions generated by each particle.
* **Analysis JSON file** containing details for each particle: perturbed image positions, softmax confidence values, maximum output values over time, and differences from the original image.

You can open `attack_analysis.json` for a detailed analysis of the attack.

---

## Documentation

Full API documentation is available at: [https://epigenomicscode.github.io/Adversarial_Observation/](https://epigenomicscode.github.io/Adversarial_Observation/)

---

## Contributing

Feel free to fork this repository and submit pull requests. Contributions are always welcome!

Please ensure any changes you propose adhere to the following guidelines:

* Write clear commit messages.
* Add or update tests as needed.
* Ensure that the code follows the existing style and conventions.

---

## Citing This Work

If you use or refer to this code in your research, please cite the following paper:


```
@incollection{gafur2024adversarial,
  title={Adversarial Robustness and Explainability of Machine Learning Models},
  author={Gafur, Jamil and Goddard, Steve and Lai, William},
  booktitle={Practice and Experience in Advanced Research Computing 2024: Human Powered Computing},
  pages={1--7},
  year={2024}
}
```

---

## License

This project is licensed under the MIT License. See the [LICENSE.txt](LICENSE.txt) file for details.
