'''
python -W ignore fig_prediagnostic_months.py
'''

import matplotlib.pyplot as plt
import argparse
import re
import os
from matplotlib.ticker import MaxNLocator


def extract_bin_edges(bins):
    edges = []
    for b in bins:
        start, end = map(int, re.findall(r'\d+', b))
        edges.append(start)
    edges.append(end)  # append last bin end
    return edges


def calculate_median_month(patients, bin_edges):
    total_patients = sum(patients)
    median_index = total_patients / 2
    cumulative = 0

    for i, count in enumerate(patients):
        if cumulative + count >= median_index:
            bin_start = bin_edges[i]
            bin_end = bin_edges[i+1]
            patients_in_bin = count
            index_in_bin = median_index - cumulative
            position_in_bin = index_in_bin / patients_in_bin
            median_month = bin_start + position_in_bin * (bin_end - bin_start)
            return median_month, i, position_in_bin
        cumulative += count

    raise ValueError("Median could not be computed.")


def plot_patient_distribution(bins, patients, figsize=(6, 6), font_size=12, linewidth=2,
                               color=(237/255, 16/255, 105/255), output_dir=".", bar_width=0.5):
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(range(len(bins)), patients, width=bar_width, color=color, edgecolor='black', linewidth=linewidth)

    ax.set_ylabel('number of patients', fontsize=font_size)
    ax.set_xlabel('months before clinical diagnosis', fontsize=font_size)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels(bins, fontsize=font_size)
    ax.tick_params(axis='y', labelsize=font_size, width=linewidth)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune=None))

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    for spine in ax.spines.values():
        spine.set_linewidth(linewidth)

    bin_edges = extract_bin_edges(bins)
    median_month, bin_index, position_in_bin = calculate_median_month(patients, bin_edges)

    # Correct x-position based on bar center location
    median_bar_x = bin_index + position_in_bin - 0.5

    ax.axvline(x=median_bar_x, color='k', linestyle='--', linewidth=linewidth,
               label=f'median = {median_month:.1f} months')

    ax.legend(fontsize=font_size, frameon=False)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'fig_prediagnostic_months.png')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot patient distribution histogram")
    parser.add_argument('--figsize', type=float, nargs=2, default=[6, 6], help='Figure size (width height)')
    parser.add_argument('--fontsize', type=int, default=20, help='Font size for labels and ticks')
    parser.add_argument('--linewidth', type=float, default=1, help='Line width for plot elements')
    parser.add_argument('--color', type=float, nargs=3, default=[237/255, 16/255, 105/255],
                        help='Bar color as RGB tuple scaled to [0,1]')
    parser.add_argument('--output_dir', type=str, default=".", help='Directory to save output figure')
    parser.add_argument('--bar_width', type=float, default=0.25, help='Width of each bar in the plot')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    bins = ['0–12', '12–24', '24–36']
    patients = [8, 2, 1]  # Approximate values from sketch
    plot_patient_distribution(
        bins,
        patients,
        figsize=tuple(args.figsize),
        font_size=args.fontsize,
        linewidth=args.linewidth,
        color=tuple(args.color),
        output_dir=args.output_dir,
        bar_width=args.bar_width
    )
