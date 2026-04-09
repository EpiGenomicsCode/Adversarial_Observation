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
# 1. Train MNIST Prediction Models
# ==========================================
echo "Starting MNIST Training..."

# Basic
singularity exec $SIF python bin/train/MNIST/basic/model_standard_MNIST_basic.py --output mnist_basic_standard.pt
singularity exec $SIF python bin/train/MNIST/basic/model_aug_MNIST_basic.py      --output mnist_basic_aug.pt
singularity exec $SIF python bin/train/MNIST/basic/model_fgsm_MNIST_basic.py     --output mnist_basic_fgsm.pt --adv-train
singularity exec $SIF python bin/train/MNIST/basic/model_pgd_MNIST_basic.py      --output mnist_basic_pgd.pt  --adv-train

# Adv
singularity exec $SIF python bin/train/MNIST/adv/model_standard_MNIST_adv.py   --output mnist_adv_standard.pt
singularity exec $SIF python bin/train/MNIST/adv/model_aug_MNIST_adv.py        --output mnist_adv_aug.pt
singularity exec $SIF python bin/train/MNIST/adv/model_fgsm_MNIST_adv.py       --output mnist_adv_fgsm.pt --adv-train
singularity exec $SIF python bin/train/MNIST/adv/model_pgd_MNIST_adv.py        --output mnist_adv_pgd.pt  --adv-train

mv mnist_*.pt models/ 2>/dev/null

# ==========================================
# 2. Train CIFAR-10 Prediction Models (E=20)
# ==========================================
echo "Starting CIFAR-10 Training..."
E=20

# Basic
singularity exec $SIF python bin/train/CIFAR10/basic/model_standard_CIFAR10_basic.py --output cifar10_basic_standard.pt --epochs $E
singularity exec $SIF python bin/train/CIFAR10/basic/model_aug_CIFAR10_basic.py      --output cifar10_basic_aug.pt      --epochs $E
singularity exec $SIF python bin/train/CIFAR10/basic/model_fgsm_CIFAR10_basic.py     --output cifar10_basic_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/CIFAR10/basic/model_pgd_CIFAR10_basic.py      --output cifar10_basic_pgd.pt      --epochs $E --adv-train

# MobileNet
singularity exec $SIF python bin/train/CIFAR10/MobileNet/model_standard_CIFAR10_MobileNet.py --output cifar10_mobilenet_standard.pt --epochs $E
singularity exec $SIF python bin/train/CIFAR10/MobileNet/model_aug_CIFAR10_MobileNet.py      --output cifar10_mobilenet_aug.pt      --epochs $E
singularity exec $SIF python bin/train/CIFAR10/MobileNet/model_fgsm_CIFAR10_MobileNet.py     --output cifar10_mobilenet_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/CIFAR10/MobileNet/model_pgd_CIFAR10_MobileNet.py      --output cifar10_mobilenet_pgd.pt      --epochs $E --adv-train

# RegNetX
singularity exec $SIF python bin/train/CIFAR10/RegNetX/model_standard_CIFAR10_RegNetX.py --output cifar10_regnetx_standard.pt --epochs $E
singularity exec $SIF python bin/train/CIFAR10/RegNetX/model_aug_CIFAR10_RegNetX.py      --output cifar10_regnetx_aug.pt      --epochs $E
singularity exec $SIF python bin/train/CIFAR10/RegNetX/model_fgsm_CIFAR10_RegNetX.py     --output cifar10_regnetx_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/CIFAR10/RegNetX/model_pgd_CIFAR10_RegNetX.py      --output cifar10_regnetx_pgd.pt      --epochs $E --adv-train

mv cifar10_*.pt models/ 2>/dev/null

# ==========================================
# 3. Train AudioMNIST Prediction Models (E=10)
# ==========================================
echo "Starting AudioMNIST Training..."
E=10

# Basic 
singularity exec $SIF python bin/train/AUDIOMNIST/basic/model_standard_AUDIOMNIST_basic.py --output audiomnist_basic_standard.pt --epochs $E
singularity exec $SIF python bin/train/AUDIOMNIST/basic/model_aug_AUDIOMNIST_basic.py      --output audiomnist_basic_aug.pt      --epochs $E
singularity exec $SIF python bin/train/AUDIOMNIST/basic/model_fgsm_AUDIOMNIST_basic.py     --output audiomnist_basic_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/AUDIOMNIST/basic/model_pgd_AUDIOMNIST_basic.py      --output audiomnist_basic_pgd.pt      --epochs $E --adv-train

# MobileNet 
singularity exec $SIF python bin/train/AUDIOMNIST/MobileNet/model_standard_AUDIOMNIST_MobileNet.py --output audiomnist_mobilenet_standard.pt --epochs $E
singularity exec $SIF python bin/train/AUDIOMNIST/MobileNet/model_aug_AUDIOMNIST_MobileNet.py      --output audiomnist_mobilenet_aug.pt      --epochs $E
singularity exec $SIF python bin/train/AUDIOMNIST/MobileNet/model_fgsm_AUDIOMNIST_MobileNet.py     --output audiomnist_mobilenet_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/AUDIOMNIST/MobileNet/model_pgd_AUDIOMNIST_MobileNet.py      --output audiomnist_mobilenet_pgd.pt      --epochs $E --adv-train

# RegNetX
singularity exec $SIF python bin/train/AUDIOMNIST/RegNetX/model_standard_AUDIOMNIST_RegNetX.py --output audiomnist_regnetx_standard.pt --epochs $E
singularity exec $SIF python bin/train/AUDIOMNIST/RegNetX/model_aug_AUDIOMNIST_RegNetX.py      --output audiomnist_regnetx_aug.pt      --epochs $E
singularity exec $SIF python bin/train/AUDIOMNIST/RegNetX/model_fgsm_AUDIOMNIST_RegNetX.py     --output audiomnist_regnetx_fgsm.pt     --epochs $E --adv-train
singularity exec $SIF python bin/train/AUDIOMNIST/RegNetX/model_pgd_AUDIOMNIST_RegNetX.py      --output audiomnist_regnetx_pgd.pt      --epochs $E --adv-train

mv audiomnist_*.pt models/ 2>/dev/null

echo "All training complete!"