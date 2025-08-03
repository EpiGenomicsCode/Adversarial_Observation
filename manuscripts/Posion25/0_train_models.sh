#!/bin/bash

module load anaconda3_gpu

mkdir -p models

DATASET="MNIST"
BATCH_SIZE=32
EPOCHS=5
SAVE_DIR="models"

# Define model types and adversarial modes
MODEL_TYPES=("normal" "complex" "complex_augmented" "complex_adversarial")
# For adversarial modes, only complex_adversarial uses them, others use "none"
ADVERSARIAL_MODES=("none" "pgd" "trades")

for model_type in "${MODEL_TYPES[@]}"; do
    if [[ "$model_type" == "complex_adversarial" ]]; then
        for adv in "${ADVERSARIAL_MODES[@]:1}"; do
            echo "Training $DATASET with model_type=$model_type and adversarial=$adv"
            echo "Command: python 0_trainModel.py --data $DATASET --model_type $model_type --adversarial $adv --batch_size $BATCH_SIZE --epochs $EPOCHS --save_dir $SAVE_DIR"
            # python 0_trainModel.py --data $DATASET --model_type $model_type --adversarial $adv --batch_size $BATCH_SIZE --epochs $EPOCHS --save_dir $SAVE_DIR
        done
    else
        echo "Training $DATASET with model_type=$model_type (no adversarial)"
        echo "Command: python 0_trainModel.py --data $DATASET --model_type $model_type --batch_size $BATCH_SIZE --epochs $EPOCHS --save_dir $SAVE_DIR"
        # python 0_trainModel.py --data $DATASET --model_type $model_type --batch_size $BATCH_SIZE --epochs $EPOCHS --save_dir $SAVE_DIR
    fi
done

echo "Training completed. Models saved under $SAVE_DIR/ directory."

exit
