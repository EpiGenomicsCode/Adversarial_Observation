import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, cifar10

def standardize_mnist(x):
    return x / 255.0

def standardize_cifar(x):
    return x / 255.0

def load_data(dataset_name, batch_size=32):
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = standardize_mnist(x_train.reshape(-1, 28, 28, 1).astype('float32'))
        x_test = standardize_mnist(x_test.reshape(-1, 28, 28, 1).astype('float32'))
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = standardize_cifar(x_train.astype('float32'))
        x_test = standardize_cifar(x_test.astype('float32'))
    else:
        raise ValueError("Unsupported dataset. Choose 'mnist' or 'cifar10'.")

    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

    return train_dataset, test_dataset

def write_labels_to_tsv(dataset, filename):
    with open(filename, 'w') as f:
        f.write("Index\tLabel\n")
        for idx, (_, label) in enumerate(dataset.unbatch()):
            true_label = int(np.argmax(label.numpy()))
            f.write(f"{idx}\t{true_label}\n")
    print(f"Wrote labels to {filename}")

def generate_false_labels(df, num_classes=10, seed=None):
    if seed is not None:
        np.random.seed(seed)

    false_labels = []
    for true_label in df['Label']:
        choices = [i for i in range(num_classes) if i != true_label]
        false_label = np.random.choice(choices)
        false_labels.append(false_label)

    df_out = pd.DataFrame({
        'Index': df['Index'],
        'trueLabel': df['Label'],
        'falseLabel': false_labels
    })
    return df_out

def main():
    parser = argparse.ArgumentParser(description="Load dataset, write labels, and optionally generate false labels TSV.")
    parser.add_argument('--dataset', type=str, required=True, choices=['mnist', 'cifar10'], help='Dataset to load.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for dataset.')
    parser.add_argument('--output_prefix', type=str, default='output', help='Prefix/path for output TSV files.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for false label generation.')
    parser.add_argument('--generate_false_labels', action='store_true', help='Generate false labels TSV after label files are created.')
    args = parser.parse_args()

    train_dataset, test_dataset = load_data(args.dataset, batch_size=args.batch_size)

    train_labels_file = f"{args.output_prefix}_{args.dataset}_train_labels.tsv"
    test_labels_file = f"{args.output_prefix}_{args.dataset}_test_labels.tsv"

    write_labels_to_tsv(train_dataset, train_labels_file)
    write_labels_to_tsv(test_dataset, test_labels_file)

    if args.generate_false_labels:
        # Read back the label files and generate false labels
        for split in ['train', 'test']:
            label_file = f"{args.output_prefix}_{args.dataset}_{split}_labels.tsv"
            df_labels = pd.read_csv(label_file, sep='\t')
            df_false = generate_false_labels(df_labels, seed=args.seed)
            false_label_file = f"{args.output_prefix}_{args.dataset}_{split}_false_labels.tsv"
            df_false.to_csv(false_label_file, sep='\t', index=False)
            print(f"Generated false labels TSV at {false_label_file}")

if __name__ == '__main__':
    main()
