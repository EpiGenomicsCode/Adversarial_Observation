#!/bin/bash

SIF=/storage/group/bfp2/default/wkl2-WillLai/Adversarial_Project/Adversarial_Observation/manuscripts/POISON25/pytorch-captum.sif
WORKINGDIR=/storage/group/bfp2/default/wkl2-WillLai/Adversarial_Project/Adversarial_Observation/manuscripts/POISON25

# ==========================================
# CONFIGURATION
# Set the model and its architecture here
# Available architectures: basic, adv, MobileNet, RegNetX
# ==========================================
MODEL_NAME="cifar10_mobilenet_standard"
ARCH="MobileNet"
MODEL_FILE="${MODEL_NAME}.pt"

OUTPUT=$WORKINGDIR/CIFAR10_test_${MODEL_NAME}
mkdir -p $OUTPUT
cd $OUTPUT

HEADER="#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --ntasks=4\n#SBATCH --mem=24GB\n#SBATCH --time=36:00:00\n#SBATCH --partition=open\n"

# Label file
LABELS=$WORKINGDIR/labels/CIFAR10_test_labels-misclassify.tsv 

# Attack script
POISON=$WORKINGDIR/bin/attack/poison_CIFAR10.py

EPOCH=30
PARTICLE=500
RETRY=1

COHORT_ID=0
COHORT_INDEX=0

echo "Preparing cohort: $COHORT_ID for model: $MODEL_NAME (Arch: $ARCH)"
echo -e $HEADER > $OUTPUT/attack_$COHORT_ID.slurm
echo "cd $OUTPUT" >> $OUTPUT/attack_$COHORT_ID.slurm

# Read the file line by line
while read line; do
    if [[ "$line" == index* ]]; then
        continue
    fi

    index=$(echo "$line" | awk '{print $1}')
    trueLabel=$(echo "$line" | awk '{print $2}')
    falseLabel=$(echo "$line" | awk '{print $3}')

    # Execute poisoning (Now passing --arch)
    echo "singularity exec -B $WORKINGDIR/models:/models $SIF bash -c \"time python $POISON --modelPath /models/$MODEL_FILE --arch $ARCH --epochs $EPOCH --particleNum $PARTICLE --maxRetries $RETRY --outputPath CIFAR10_test_$index --targetLabel $falseLabel --sourceIndex $index\"" >> $OUTPUT/attack_$COHORT_ID.slurm
    ((COHORT_INDEX++))

    if [ $COHORT_INDEX -gt 100 ]; then
        COHORT_INDEX=0
        ((COHORT_ID++))

        echo "Preparing cohort: $COHORT_ID"
        echo -e $HEADER > $OUTPUT/attack_$COHORT_ID.slurm
        echo "cd $OUTPUT" >> $OUTPUT/attack_$COHORT_ID.slurm
    fi

done < "$LABELS"