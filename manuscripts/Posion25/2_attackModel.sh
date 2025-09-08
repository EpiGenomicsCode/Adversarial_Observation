#!/bin/bash

# Base working directory
WORKINGDIR=/workspaces/Adversarial_Observation/manuscripts/Posion25

# Label and model paths
LABELDIR=$WORKINGDIR/false_data
MODELDIR=$WORKINGDIR/models/MNIST_normal
MODEL=$MODELDIR/MNIST_normal.keras
DATA=MNIST

# Attack script (adjust path if needed)
POISON=$WORKINGDIR/2_attackModel.py

# Output directory for SLURM scripts
OUTPUT=$WORKINGDIR/slurm_jobs
mkdir -p $OUTPUT
cd $OUTPUT

# SLURM script header
HEADER="#!/bin/bash\n#SBATCH -A bbse-delta-cpu\n#SBATCH --partition=cpu\n#SBATCH --nodes=1\n#SBATCH --tasks=1\n#SBATCH --cpus-per-task=4\n#SBATCH --mem=24g\n#SBATCH --time=8:00:00\n"

# Label file to use
LABELS=$LABELDIR/MNIST_train_false_labels.tsv

# Attack parameters
ITERATION=30
PARTICLE=500

# Initialize cohort tracking
COHORT_ID=0
COHORT_INDEX=0

# Start first SLURM script
echo "Preparing cohort: $COHORT_ID"
echo -e $HEADER > $OUTPUT/attack_$COHORT_ID.slurm
echo "module load anaconda3_gpu" >> $OUTPUT/attack_$COHORT_ID.slurm
echo "cd $OUTPUT" >> $OUTPUT/attack_$COHORT_ID.slurm

# Read through label file
while read line; do
    # Skip header
    if [[ "$line" == index* ]]; then
        continue
    fi

    # Parse line
    index=$(echo "$line" | awk '{print $1}')
    trueLabel=$(echo "$line" | awk '{print $2}')
    falseLabel=$(echo "$line" | awk '{print $3}')

    # Add attack command
    echo "time python $POISON --data $DATA --model_path $MODEL --iterations $ITERATION --particles $PARTICLE --save_dir MNIST_train_$index --target $falseLabel --source_index $index" >> $OUTPUT/attack_$COHORT_ID.slurm

    ((COHORT_INDEX++))

    if [ $COHORT_INDEX -gt 100 ]; then
        COHORT_INDEX=0
        ((COHORT_ID++))

        echo "Preparing cohort: $COHORT_ID"
        echo -e $HEADER > $OUTPUT/attack_$COHORT_ID.slurm
        echo "module load anaconda3_gpu" >> $OUTPUT/attack_$COHORT_ID.slurm
        echo "cd $OUTPUT" >> $OUTPUT/attack_$COHORT_ID.slurm
    fi

done < "$LABELS"
