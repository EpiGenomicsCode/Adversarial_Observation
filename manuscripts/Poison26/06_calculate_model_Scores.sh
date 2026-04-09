#!/bin/bash

SIF=/storage/group/bfp2/default/wkl2-WillLai/Adversarial_Project/Adversarial_Observation/manuscripts/POISON25/pytorch-captum.sif
WORKINGDIR=/storage/group/bfp2/default/wkl2-WillLai/Adversarial_Project/Adversarial_Observation/manuscripts/POISON25

# ==========================================
# CONFIGURATION
# ==========================================
DATASET="MNIST" # Change to CIFAR10 or audioMNIST as needed

mkdir -p $WORKINGDIR/${DATASET}_stats
MERGE=$WORKINGDIR/bin/infer/merge_MNIST-CSV.py
INFER=$WORKINGDIR/bin/infer/infer_MNIST.py

# The model that the adversarial examples were originally generated to fool
SOURCE_MODEL="mnist_basic_standard" 

# Array mapping: "model_name architecture"
TARGET_MODELS=(
    "mnist_basic_standard basic"
    "mnist_basic_aug basic"
    "mnist_adv_standard adv"
    "mnist_mobilenet_standard MobileNet"
    "mnist_regnetx_pgd RegNetX"
)

MERGE_INPUT_DIR="$WORKINGDIR/${DATASET}_test_${SOURCE_MODEL}_ALL/"
MERGED_CSV="$WORKINGDIR/${DATASET}_stats/${SOURCE_MODEL}-poison_ALL_best_particle_image_denoise.csv"

# ==========================================
# 1. Merge CSVs
# ==========================================
echo "Merging CSVs for attacks generated against $SOURCE_MODEL..."
python $MERGE $MERGE_INPUT_DIR --output_file $MERGED_CSV

# ==========================================
# 2. Cross-Model Inference Loop
# ==========================================
echo "Starting Inference..."

for TARGET_INFO in "${TARGET_MODELS[@]}"; do
    # Read the string into two separate variables
    read -r TARGET ARCH <<< "$TARGET_INFO"
    
    echo "Evaluating $TARGET ($ARCH) against $SOURCE_MODEL attacks..."
    
    MODEL_FILE="${TARGET}.pt"
    OUTFILE="$WORKINGDIR/${DATASET}_stats/${SOURCE_MODEL}-Poison_${TARGET}-Performance.tsv"
    
    # Note: Make sure your `infer_MNIST.py` script has been updated to accept --arch!
    singularity exec -B $WORKINGDIR/models:/models $SIF python $INFER \
        --model /models/$MODEL_FILE \
        --arch $ARCH \
        --csv $MERGED_CSV \
        --outfile $OUTFILE

done

echo "Inference scoring complete!"