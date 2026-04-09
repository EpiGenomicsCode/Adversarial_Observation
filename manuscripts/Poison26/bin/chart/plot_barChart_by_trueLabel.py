#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main(file1, file2, output):
    # --- Load and combine data ---
    df1 = pd.read_csv(file1, sep='\t')
    df2 = pd.read_csv(file2, sep='\t')
    df = pd.concat([df1, df2], ignore_index=True)

    # --- Ensure correct types ---
    df['success'] = df['success'].astype(bool)
    df['trueLabel'] = df['trueLabel'].astype(int)

    # --- Summarize success/failure counts per trueLabel ---
    summary = df.groupby(['trueLabel', 'success']).size().unstack(fill_value=0)
    summary = summary.rename(columns={True: 'Success', False: 'Failure'})

    # --- Compute percentages ---
    summary['Total'] = summary['Success'] + summary['Failure']
    summary['Success_%'] = (summary['Success'] / summary['Total']) * 100
    summary['Failure_%'] = 100 - summary['Success_%']

    # --- Prepare data for stacked percentage bar ---
    percent_df = summary[['Success_%', 'Failure_%']]

    # --- Plot ---
    ax = percent_df.plot(
        kind='bar',
        stacked=True,
        color=['#00A000', '#000000'],  # green success, black failure
        figsize=(8, 5)
    )

    plt.title('Attack Success Rate per True Label')
    plt.xlabel('True Label')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 100)
    plt.legend(title='Outcome', loc='upper right')
    plt.tight_layout()

    # --- Annotate success rate above each bar ---
    for idx, row in summary.iterrows():
        ax.text(
            idx,
            102,  # slightly above 100%
            f"{row['Success_%']:.1f}%",
            ha='center',
            va='bottom',
            fontsize=9,
            color='green'
        )

    # --- Save as SVG ---
    output = output if output else "success_rate_by_trueLabel.svg"
    plt.savefig(output, format='svg', bbox_inches='tight')
    print(f"Saved stacked percentage bar chart to: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot success percentage stacked bar chart per trueLabel.")
    parser.add_argument("file1", help="Path to first TSV file")
    parser.add_argument("file2", help="Path to second TSV file")
    parser.add_argument("-o", "--output", help="Output SVG filename (default: success_rate_by_trueLabel.svg)")
    args = parser.parse_args()

    main(args.file1, args.file2, args.output)

