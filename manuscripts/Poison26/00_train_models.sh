#!/bin/bash
#SBATCH -A bbse-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --tasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60g
#SBATCH --time=8:00:00

SIF=/work/hdd/bbse/wklai/AdversarialData/APSO_Poison/manuscripts/POISON25/pytorch-captum.sif

cd /work/hdd/bbse/wklai/AdversarialData/APSO_Poison/manuscripts/POISON25
mkdir -p models

# ==========================================
# 1. Train MNIST Prediction Models (Epochs: Default/5)
# ==========================================
echo "Starting MNIST Training..."

# Basic 2D CNN
singularity exec $SIF python bin/train/MNIST/basic/model_standard.py --output mnist_basic_standard.pt
singularity exec $SIF python bin/train/MNIST/basic/model_aug.py      --output mnist_basic_aug.pt
singularity exec $SIF python bin/train/MNIST/basic/model_fgsm.py     --output mnist_basic_fgsm.pt --adv-train
singularity exec $SIF python bin/train/MNIST/basic/model_pgd.py      --output mnist_basic_pgd.pt  --adv-train

# Advanced 2D CNN (VGG-style)
singularity exec $SIF python bin/train/MNIST/adv/model_standard.py   --output mnist_adv_standard.pt
singularity exec $SIF python bin/train/MNIST/adv/model_aug.py        --output mnist_adv_aug.pt
singularity exec $SIF python bin/train/MNIST/adv/model_fgsm.py       --output mnist_adv_fgsm.pt --adv-train
singularity exec $SIF python bin/train/MNIST/adv/model_pgd.py        --output mnist_adv_pgd.pt  --adv-train

mv mnist_*.pt models/


# ==========================================
# 2. Train CIFAR-10 Prediction Models (Epochs: 20)
# ==========================================
echo "Starting CIFAR-10 Training..."
E=20

# Basic 2D CNN
singularity exec $SIF python bin/train/CIFAR10/basic/model_standard.py --output cifar10_basic_standard.pt --epochs $E
singularity exec $SIF python bin/train/CIFAR10/basic/model_aug.py      --output cifar10_basic_aug.pt      --epochs $E
singularity exec $SIF python bin/train/CIFAR10/basic/model_fgsm.py     --output cifar10_basic_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/CIFAR10/basic/model_pgd.py      --output cifar10_basic_pgd.pt      --epochs $E --adv-train

# MobileNet
singularity exec $SIF python bin/train/CIFAR10/MobileNet/model_standard.py --output cifar10_mobilenet_standard.pt --epochs $E
singularity exec $SIF python bin/train/CIFAR10/MobileNet/model_aug.py      --output cifar10_mobilenet_aug.pt      --epochs $E
singularity exec $SIF python bin/train/CIFAR10/MobileNet/model_fgsm.py     --output cifar10_mobilenet_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/CIFAR10/MobileNet/model_pgd.py      --output cifar10_mobilenet_pgd.pt      --epochs $E --adv-train

# RegNetX
singularity exec $SIF python bin/train/CIFAR10/RegNetX/model_standard.py --output cifar10_regnetx_standard.pt --epochs $E
singularity exec $SIF python bin/train/CIFAR10/RegNetX/model_aug.py      --output cifar10_regnetx_aug.pt      --epochs $E
singularity exec $SIF python bin/train/CIFAR10/RegNetX/model_fgsm.py     --output cifar10_regnetx_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/CIFAR10/RegNetX/model_pgd.py      --output cifar10_regnetx_pgd.pt      --epochs $E --adv-train

mv cifar10_*.pt models/


# ==========================================
# 3. Train AudioMNIST Prediction Models (Epochs: 10)
# ==========================================
echo "Starting AudioMNIST Training..."
E=10

# Basic 1D CNN
singularity exec $SIF python bin/train/AudioMNIST/basic/model_standard.py --output audiomnist_basic_standard.pt --epochs $E
singularity exec $SIF python bin/train/AudioMNIST/basic/model_aug.py      --output audiomnist_basic_aug.pt      --epochs $E
singularity exec $SIF python bin/train/AudioMNIST/basic/model_fgsm.py     --output audiomnist_basic_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/AudioMNIST/basic/model_pgd.py      --output audiomnist_basic_pgd.pt      --epochs $E --adv-train

# MobileNet (2D Spectrogram wrapper)
singularity exec $SIF python bin/train/AudioMNIST/MobileNet/model_standard.py --output audiomnist_mobilenet_standard.pt --epochs $E
singularity exec $SIF python bin/train/AudioMNIST/MobileNet/model_aug.py      --output audiomnist_mobilenet_aug.pt      --epochs $E
singularity exec $SIF python bin/train/AudioMNIST/MobileNet/model_fgsm.py     --output audiomnist_mobilenet_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/AudioMNIST/MobileNet/model_pgd.py      --output audiomnist_mobilenet_pgd.pt      --epochs $E --adv-train

mv audiomnist_*.pt models/

echo "All training complete!"