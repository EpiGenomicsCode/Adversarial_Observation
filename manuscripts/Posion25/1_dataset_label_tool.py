import argparse
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.utils import to_categorical

# -------------------- Utility Functions --------------------

def generate_false_labels(df, seed=None):
    if seed is not None:
        np.random.seed(seed)

    num_classes = 10
    false_labels = []

    for true_label in df['Label']:
        choices = [i for i in range(num_classes) if i != true_label]
        false_label = np.random.choice(choices)
        false_labels.append(false_label)

    df_out = pd.DataFrame({
        'index': df['Index'],
        'trueLabel': df['Label'],
        'falseLabel': false_labels
    })
    return df_out

def standardize_data(x):
    """Normalize data to [0, 1] range."""
    return x / 255.0

def write_labels_to_tsv(dataset, output_file):
    with open(output_file, 'w') as f:
        f.write("Index\tLabel\n")
        for idx, (_, label) in enumerate(dataset.unbatch()):
            true_label = int(np.argmax(label.numpy()))
            f.write(f"{idx}\t{true_label}\n")
    print(f"Wrote labels to {output_file}")

# -------------------- Dataset Loaders --------------------

def load_mnist(batch_size):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = standardize_data(x_train.reshape(-1, 28, 28, 1).astype('float32'))
    x_test = standardize_data(x_test.reshape(-1, 28, 28, 1).astype('float32'))
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_dataset, test_dataset

def load_cifar10(batch_size):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = standardize_data(x_train.astype('float32'))
    x_test = standardize_data(x_test.astype('float32'))
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)
    return train_dataset, test_dataset

# -------------------- Main Script --------------------

def main():
    parser = argparse.ArgumentParser(description="Dataset label handler.")
    parser.add_argument('--dataset', choices=['mnist', 'cifar10'], help="Dataset to process.")
    parser.add_argument('--generate', action='store_true', help="Generate TSV files from dataset.")
    parser.add_argument('--input', help="Input TSV file (for false label generation).")
    parser.add_argument('--output', help="Output TSV file name (saved in false_data/).")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for data loading.")

    args = parser.parse_args()

    # Create folder for false labels
    false_data_dir = "false_data"
    os.makedirs(false_data_dir, exist_ok=True)

    # Generate TSV files from the dataset
    if args.generate:
        if args.dataset == 'mnist':
            train_ds, test_ds = load_mnist(args.batch_size)
            write_labels_to_tsv(train_ds, os.path.join(false_data_dir, "MNIST_train_labels.tsv"))
            write_labels_to_tsv(test_ds, os.path.join(false_data_dir, "MNIST_test_labels.tsv"))
        elif args.dataset == 'cifar10':
            train_ds, test_ds = load_cifar10(args.batch_size)
            write_labels_to_tsv(train_ds, os.path.join(false_data_dir, "CIFAR10_train_labels.tsv"))
            write_labels_to_tsv(test_ds, os.path.join(false_data_dir, "CIFAR10_test_labels.tsv"))
        else:
            print("Please specify a valid dataset with --dataset.")

    # Generate false labels from input TSV
    if args.input and args.output:
        input_path = args.input
        if not os.path.exists(input_path):
            input_path = os.path.join(false_data_dir, args.input)

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")

        df = pd.read_csv(input_path, sep='\t')
        df_out = generate_false_labels(df, seed=args.seed)

        output_path = os.path.join(false_data_dir, args.output)
        df_out.to_csv(output_path, sep='\t', index=False)
        print(f"False labels saved to {output_path}")

if __name__ == '__main__':
    main()
