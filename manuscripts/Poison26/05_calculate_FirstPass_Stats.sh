SIF=/storage/group/bfp2/default/wkl2-WillLai/Adversarial_Project/Adversarial_Observation/manuscripts/POISON25/pytorch-captum.sif

WORKINGDIR=/storage/group/bfp2/default/wkl2-WillLai/Adversarial_Project/Adversarial_Observation/manuscripts/POISON25
mkdir -p $WORKINGDIR/MNIST_stats

LABELS=labels/MNIST_test_labels-misclassify.tsv

EVAL=$WORKINGDIR/bin/eval/evaluate_poisoning.py
PARSE=$WORKINGDIR/bin/eval/convert_results.py

MODEL=model1
singularity exec $SIF python $EVAL --main_folder $WORKINGDIR/MNIST_test_$MODEL --labels_file $LABELS  --output_prefix $WORKINGDIR/MNIST_stats/MNIST_stats-$MODEL
singularity exec $SIF python $PARSE --input_file $WORKINGDIR/MNIST_stats/MNIST_stats-$MODEL\_fail.tsv --output_file $WORKINGDIR/MNIST_stats/MNIST_test_labels_$MODEL\-misclassify_RESILIENT.tsv

MODEL=model2
singularity exec $SIF python $EVAL --main_folder $WORKINGDIR/MNIST_test_$MODEL --labels_file $LABELS  --output_prefix $WORKINGDIR/MNIST_stats/MNIST_stats-$MODEL
singularity exec $SIF python $PARSE --input_file $WORKINGDIR/MNIST_stats/MNIST_stats-$MODEL\_fail.tsv --output_file $WORKINGDIR/MNIST_stats/MNIST_test_labels_$MODEL\-misclassify_RESILIENT.tsv

MODEL=model3
singularity exec $SIF python $EVAL --main_folder $WORKINGDIR/MNIST_test_$MODEL --labels_file $LABELS  --output_prefix $WORKINGDIR/MNIST_stats/MNIST_stats-$MODEL
singularity exec $SIF python $PARSE --input_file $WORKINGDIR/MNIST_stats/MNIST_stats-$MODEL\_fail.tsv --output_file $WORKINGDIR/MNIST_stats/MNIST_test_labels_$MODEL\-misclassify_RESILIENT.tsv

MODEL=model4
singularity exec $SIF python $EVAL --main_folder $WORKINGDIR/MNIST_test_$MODEL --labels_file $LABELS  --output_prefix $WORKINGDIR/MNIST_stats/MNIST_stats-$MODEL
singularity exec $SIF python $PARSE --input_file $WORKINGDIR/MNIST_stats/MNIST_stats-$MODEL\_fail.tsv --output_file $WORKINGDIR/MNIST_stats/MNIST_test_labels_$MODEL\-misclassify_RESILIENT.tsv

MODEL=model5
singularity exec $SIF python $EVAL --main_folder $WORKINGDIR/MNIST_test_$MODEL --labels_file $LABELS  --output_prefix $WORKINGDIR/MNIST_stats/MNIST_stats-$MODEL
singularity exec $SIF python $PARSE --input_file $WORKINGDIR/MNIST_stats/MNIST_stats-$MODEL\_fail.tsv --output_file $WORKINGDIR/MNIST_stats/MNIST_test_labels_$MODEL\-misclassify_RESILIENT.tsv

