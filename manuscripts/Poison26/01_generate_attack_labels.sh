module load anaconda3_cpu

mkdir -p labels
LABEL=bin/utils/generate_FalseLabels.py

OUTPUT=bin/utils/output_MNIST_labels.py
python $OUTPUT
mv MNIST*labels.tsv labels/
python $LABEL --input labels/MNIST_test_labels.tsv --output labels/MNIST_test_labels-misclassify.tsv --seed 1

OUTPUT=bin/utils/output_CIFAR10_labels.py

python $OUTPUT
mv CIFAR10*labels.tsv labels/
python $LABEL --input labels/CIFAR10_test_labels.tsv --output labels/CIFAR10_test_labels-misclassify.tsv --seed 1
