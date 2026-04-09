import argparse
import pandas as pd
import numpy as np

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

def main():
    parser = argparse.ArgumentParser(description='Generate false labels TSV.')
    parser.add_argument('--input', required=True, help='Input TSV file')
    parser.add_argument('--output', required=True, help='Output TSV file')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility')
    args = parser.parse_args()

    df = pd.read_csv(args.input, sep='\t')
    df_out = generate_false_labels(df, seed=args.seed)
    df_out.to_csv(args.output, sep='\t', index=False)

if __name__ == '__main__':
    main()

