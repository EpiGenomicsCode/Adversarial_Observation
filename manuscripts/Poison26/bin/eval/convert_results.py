import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Convert poisoning results back to original label TSV format.")
    parser.add_argument("--input_file", type=str, required=True,
                        help="Path to poisoning results TSV (success or fail).")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to output TSV in original format (index, trueLabel, falseLabel).")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load results
    df = pd.read_csv(args.input_file, sep="\t")

    # Keep only original columns
    output_df = df[["index", "trueLabel", "falseLabel"]]

    # Save in original format
    output_df.to_csv(args.output_file, sep="\t", index=False)

    print(f"Converted {args.input_file} -> {args.output_file}")


if __name__ == "__main__":
    main()

