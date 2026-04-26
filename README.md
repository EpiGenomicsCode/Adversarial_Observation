# Adversarial Observation

A framework for black-box adversarial poisoning attacks using **Particle Swarm Optimization (PSO)**, with analysis of attack resilience across model architectures and adversarial training regimes.

The current manuscript (in preparation) evaluates PSO-based poisoning attacks against models trained on **MNIST**, **CIFAR-10**, and **AudioMNIST**, examining how adversarial training (FGSM, PGD) and architecture choice (CNN, MobileNet, RegNetX) affect resilience, and whether poisoning transfers across model families.

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Setup and Installation](#setup-and-installation)
4. [Singularity Container](#singularity-container)
   * [Build](#build)
   * [Run](#run)
5. [Testing](#testing)
6. [Package Structure](#package-structure)
7. [Experimental Design](#experimental-design)
8. [Pipeline](#pipeline)
   * [Step 0 — Train Models](#step-0--train-models)
   * [Step 1 — Generate Attack Labels](#step-1--generate-attack-labels)
   * [Steps 2–4 — Run PSO Attacks](#steps-24--run-pso-attacks)
   * [Steps 5–6 — Aggregate Results](#steps-56--aggregate-results)
9. [Directory Structure](#directory-structure)
10. [Documentation](#documentation)
11. [Contributing](#contributing)
12. [Citing This Work](#citing-this-work)
13. [License](#license)

---

## Overview

**Adversarial Observation** provides a PSO-based black-box adversarial attack framework built on PyTorch. The swarm optimizer searches the input space to find minimal perturbations that cause a target model to misclassify an input as a chosen false label, without requiring gradient access.

The Poison26 experiments systematically measure:

* **Resilience** — how adversarial training strategies (standard, FGSM, PGD) affect susceptibility to PSO-based poisoning
* **Transfer** — whether poisoning attacks that succeed against one architecture generalize to others trained on the same data

---

## Requirements

* Python 3.10+
* PyTorch 2.4.1 / torchvision 0.19.1 / torchaudio 2.4.1
* numpy, scipy, matplotlib, scikit-learn, pandas, imageio, librosa
* captum

For HPC / reproducible runs, use the provided Singularity container (see below). For local development:

```bash
pip install -r requirements.txt
```

---

## Setup and Installation

```bash
git clone https://github.com/EpiGenomicsCode/Adversarial_Observation.git
cd Adversarial_Observation
pip install -e .
```

---

## Singularity Container

A Singularity definition file is provided at [singularity/apso_poison.def](singularity/apso_poison.def). It builds a `pytorch-captum` conda environment and installs the `Adversarial_Observation` package into it.

### Build

Run from the **repository root** — the `%files` directive copies `setup.py` and the `Adversarial_Observation/` package relative to the current directory:

```bash
singularity build apso_poison.sif singularity/apso_poison.def
```

### Run

Pass any command as arguments — the container executes it inside the `pytorch-captum` environment:

```bash
singularity exec apso_poison.sif python manuscripts/Poison26/bin/train/MNIST/train_MNIST.py \
    --arch basic --training standard --output mnist_basic_standard.pt
```

The SLURM runbooks in `manuscripts/Poison26/` reference the container via `$SIF` and are ready to submit directly to an A100 GPU partition.

---

## Testing

Unit tests cover the PSO optimizer, individual particles, adversarial attacks, and data loading. All tests require PyTorch; run them inside the Singularity container or any environment where the package is installed.

**Inside the container:**

```bash
singularity exec apso_poison.sif python -m pytest tests/ -v
```

**In a local conda/venv environment:**

```bash
pip install -e .
pytest tests/ -v
```

The `test_apso_singularity.py` suite specifically validates the APSO workflow as used by the Poison26 attack scripts (initialization, step invariants, position clamping, and captum availability). The captum test is automatically skipped if captum is not installed in the local environment.

---

## Package Structure

The `Adversarial_Observation` package:

| Module | Purpose |
|---|---|
| `Swarm.py` | PSO orchestration — runs particles across iterations, tracks global best |
| `BirdParticle.py` | Individual particle: position, velocity, personal best |
| `Attacks.py` | FGSM, gradient ascent, gradient maps, saliency maps |
| `utils.py` | Data loading, model loading, metrics, seed utilities |
| `visualize.py` | GIF generation from per-iteration attack frames |

---

## Experimental Design

| Dataset | Architectures | Training Regimes |
|---|---|---|
| MNIST | basic CNN, adv CNN, MobileNet, RegNetX | standard, FGSM, PGD (± augmentation) |
| CIFAR-10 | basic CNN, adv CNN, MobileNet, RegNetX | standard, FGSM, PGD (± augmentation) |
| AudioMNIST | basic CNN, adv CNN, MobileNet, RegNetX | standard, FGSM, PGD (± augmentation) |

Each model variant is trained, then subjected to PSO-based poisoning attacks targeting each possible misclassification label. Results are aggregated to produce per-model resilience scores and cross-model transfer statistics.

---

## Pipeline

All steps are SLURM-ready scripts under `manuscripts/Poison26/`. Set `$SIF` to your built `apso_poison.sif` path before running.

### Step 0 — Train Models

```bash
sbatch manuscripts/Poison26/00_train_models.sh
```

Trains all architecture × training-regime combinations for MNIST, CIFAR-10, and AudioMNIST. Model weights are saved to `models/{MNIST,CIFAR10,AUDIOMNIST}/`.

### Step 1 — Generate Attack Labels

```bash
sbatch manuscripts/Poison26/01_generate_attack_labels.sh
```

Exports true labels for each dataset's test split and generates false-label targets (`labels/*_labels-misclassify.tsv`) used as PSO attack objectives.

### Steps 2–4 — Run PSO Attacks

```bash
sbatch manuscripts/Poison26/02_attack_MNIST_Models-Resilience.sh
sbatch manuscripts/Poison26/03_attack_CIFAR10_Models-Resilience.sh
sbatch manuscripts/Poison26/04_attack_audioMNIST_Models-Resilience.sh
```

For each dataset, runs `bin/attack/poison_<dataset>.py` against every trained model variant, saving per-sample attack results (best perturbation, confidence trajectory, outcome) to TSV files.

### Steps 5–6 — Aggregate Results

```bash
sbatch manuscripts/Poison26/05_calculate_FirstPass_Stats.sh
sbatch manuscripts/Poison26/06_calculate_model_Scores.sh
```

Computes first-iteration success rates and aggregate resilience scores. Visualization scripts in `bin/chart/` generate violin and bar plots for cross-model and cross-label comparisons.

---

## Directory Structure

```
Adversarial_Observation/       # installable Python package
singularity/
└── apso_poison.def            # Singularity container definition (build from repo root)
manuscripts/
├── PEARC24/                   # companion code for the published PEARC'24 paper
└── Poison26/                  # current manuscript experiments
    ├── bin/
    │   ├── attack/            # PSO poisoning scripts (MNIST, CIFAR10, AudioMNIST)
    │   ├── chart/             # violin and bar chart generators
    │   ├── eval/              # attack success evaluation
    │   ├── infer/             # result aggregation / CSV merging
    │   ├── train/             # model training scripts by dataset and architecture
    │   └── utils/             # label export and model evaluation utilities
    ├── labels/                # generated true/false label TSV files
    ├── models/                # trained model checkpoints (not committed)
    ├── 00_train_models.sh
    ├── 01_generate_attack_labels.sh
    ├── 02_attack_MNIST_Models-Resilience.sh
    ├── 03_attack_CIFAR10_Models-Resilience.sh
    ├── 04_attack_audioMNIST_Models-Resilience.sh
    ├── 05_calculate_FirstPass_Stats.sh
    └── 06_calculate_model_Scores.sh
tests/                         # unit tests for PSO, particle, attacks, data loading
docs/                          # Sphinx API documentation source
```

---

## Documentation

Full API documentation: [https://epigenomicscode.github.io/Adversarial_Observation/](https://epigenomicscode.github.io/Adversarial_Observation/)

---

## Contributing

Pull requests are welcome. Please:

* Write clear commit messages
* Add or update tests as needed
* Follow existing code style and conventions

---

## Citing This Work

If you use this code, please cite the published PEARC'24 paper:

```bibtex
@incollection{gafur2024adversarial,
  title={Adversarial Robustness and Explainability of Machine Learning Models},
  author={Gafur, Jamil and Goddard, Steve and Lai, William},
  booktitle={Practice and Experience in Advanced Research Computing 2024: Human Powered Computing},
  pages={1--7},
  year={2024}
}
```

A manuscript describing the Poison26 experiments is currently in preparation.

---

## License

This project is licensed under the MIT License. See [LICENSE.txt](LICENSE.txt) for details.
