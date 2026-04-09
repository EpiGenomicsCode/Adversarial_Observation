#!/bin/bash

SIF=/storage/group/bfp2/default/wkl2-WillLai/Adversarial_Project/Adversarial_Observation/manuscripts/POISON25/pytorch-captum.sif

WORKINGDIR=/storage/group/bfp2/default/wkl2-WillLai/Adversarial_Project/Adversarial_Observation/manuscripts/POISON25
OUTPUT=$WORKINGDIR/MNIST_test_model4
mkdir -p $OUTPUT
cd $OUTPUT

#HEADER="#!/bin/bash\n#SBATCH -A bbse-delta-cpu\n#SBATCH --partition=cpu\n#SBATCH --nodes=1\n#SBATCH --tasks=1\n#SBATCH --cpus-per-task=4\n#SBATCH --mem=24g\n#SBATCH --time=8:00:00\n"
HEADER="#!/bin/bash\n#SBATCH --nodes=1\n#SBATCH --ntasks=4\n#SBATCH --mem=24GB\n#SBATCH --time=36:00:00\n#SBATCH --partition=open\n"

# Label file
#LABELS=$WORKINGDIR/labels/MNIST_test_labels-misclassify_r2.tsv
#LABELS=$WORKINGDIR/labels/MNIST_test_labels-misclassify.tsv
LABELS=$WORKINGDIR/MNIST_stats/MNIST_test_labels_model4-misclassify_RESILIENT.tsv
# Model file
MODEL=MNIST_model4.pt
# Attack script
POISON=$WORKINGDIR/bin/attack/poison_MNIST.py

EPOCH=30
PARTICLE=500
RETRY=5

COHORT_ID=0
COHORT_INDEX=0

echo "Preparing cohort: $COHORT_ID"
echo -e $HEADER > $OUTPUT/attack_$COHORT_ID.slurm
# Delta-specific
#echo "module load anaconda3_gpu" >> $OUTPUT/attack_$COHORT_ID.slurm

echo "cd $OUTPUT" >> $OUTPUT/attack_$COHORT_ID.slurm

# Read the file line by line
while read line; do
    # Skip the header line
    if [[ "$line" == index* ]]; then
        continue
    fi
#    echo "Preparing cohort: $COHORT_ID"

    # Extract values using awk
    index=$(echo "$line" | awk '{print $1}')
    trueLabel=$(echo "$line" | awk '{print $2}')
    falseLabel=$(echo "$line" | awk '{print $3}')

    # Use these variables as needed
    echo "singularity exec -B $WORKINGDIR/models:/models $SIF bash -c \"time python $POISON --modelPath /models/$MODEL --epochs $EPOCH --particleNum $PARTICLE --maxRetries $RETRY --outputPath MNIST_test_$index --targetLabel $falseLabel --sourceIndex $index\"" >> $OUTPUT/attack_$COHORT_ID.slurm
    ((COHORT_INDEX++))

    if [ $COHORT_INDEX -gt 70 ]; then
	COHORT_INDEX=0
	((COHORT_ID++))

	echo "Preparing cohort: $COHORT_ID"
	echo -e $HEADER > $OUTPUT/attack_$COHORT_ID.slurm
#	echo "module load anaconda3_gpu" >> $OUTPUT/attack_$COHORT_ID.slurm

	echo "cd $OUTPUT" >> $OUTPUT/attack_$COHORT_ID.slurm
    fi

done < "$LABELS"

