#!/bin/bash

# Load environment
module load anaconda3_cpu

# Create output directory
mkdir -p labels

# Paths to scripts
LABEL_SCRIPT="bin/utils/generate_FalseLabels.py"
OUTPUT_SCRIPTS=("bin/utils/output_MNIST_labels.py" "bin/utils/output_CIFAR10_labels.py")

# Loop through each dataset output script
for OUTPUT_SCRIPT in "${OUTPUT_SCRIPTS[@]}"; do
    python "$OUTPUT_SCRIPT"
    
    # Extract dataset name (e.g., MNIST or CIFAR10)
    DATASET=$(basename "$OUTPUT_SCRIPT" | cut -d'_' -f2)

    # Move generated label files
    mv ${DATASET}*labels.tsv labels/

    # Generate false labels
    python "$LABEL_SCRIPT" \
        --input "labels/${DATASET}_test_labels.tsv" \
        --output "labels/${DATASET}_test_labels-misclassify.tsv" \
        --seed 1
done
