#!/bin/bash

# Load environment (or replace with your Singularity execution if dependencies require it)
module load anaconda3_cpu

# Ensure the output directory exists
mkdir -p labels
LABEL=bin/utils/generate_FalseLabels.py

# ==========================================
# 1. Generate MNIST Labels
# ==========================================
echo "Generating MNIST attack labels..."
OUTPUT=bin/utils/output_MNIST_labels.py
python $OUTPUT

# Move generated files and create misclassification targets
mv MNIST*labels.tsv labels/ 2>/dev/null
python $LABEL --input labels/MNIST_test_labels.tsv --output labels/MNIST_test_labels-misclassify.tsv --seed 1


# ==========================================
# 2. Generate CIFAR-10 Labels
# ==========================================
echo "Generating CIFAR-10 attack labels..."
OUTPUT=bin/utils/output_CIFAR10_labels.py
python $OUTPUT

# Move generated files and create misclassification targets
mv CIFAR10*labels.tsv labels/ 2>/dev/null
python $LABEL --input labels/CIFAR10_test_labels.tsv --output labels/CIFAR10_test_labels-misclassify.tsv --seed 1


# ==========================================
# 3. Generate AudioMNIST Labels
# ==========================================
echo "Generating AudioMNIST attack labels..."
OUTPUT=bin/utils/output_audioMNIST_labels.py

# Added a safety check in case the utility script has a slightly different naming convention
if [ -f "$OUTPUT" ]; then
    python $OUTPUT
    # Catch both camelCase and lowercase generations just in case
    mv audioMNIST*labels.tsv labels/ 2>/dev/null || mv AudioMNIST*labels.tsv labels/ 2>/dev/null
    
    python $LABEL --input labels/audioMNIST_test_labels.tsv --output labels/audioMNIST_test_labels-misclassify.tsv --seed 1
else
    echo "Warning: $OUTPUT not found. Skipping AudioMNIST label generation. Ensure the utility script exists."
fi

echo "All attack labels successfully generated!"