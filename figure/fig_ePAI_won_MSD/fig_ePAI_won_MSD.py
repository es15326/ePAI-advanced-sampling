'''
python -W ignore fig_ePAI_won_MSD.py --fontsize 12
'''

import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from matplotlib.ticker import MaxNLocator

def parse_args():
    parser = argparse.ArgumentParser(description="Plot DSC performance for different methods")
    parser.add_argument('--figsize', type=float, nargs=2, default=[4.5, 4], help='Figure size (width height)')
    parser.add_argument('--fontsize', type=int, default=14, help='Font size for labels and ticks')
    parser.add_argument('--linewidth', type=float, default=1, help='Line width for bars and spines')
    parser.add_argument('--bar_width', type=float, default=0.5, help='Width of each bar')
    parser.add_argument('--ytick_bins', type=int, default=4, help='Number of bins in the y-axis')
    parser.add_argument('--ylim', type=float, nargs=2, default=[40, 80], help='Y-axis limits (min max)')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output figure')
    parser.add_argument('--axiswidth', type=float, default=1.2, help='Width of the axes')
    parser.add_argument('--labelpad', type=int, default=5, help='Padding for labels')
    return parser.parse_args()

def plot_performance_bar(args):
    methods = [
        "Kim et al.", "C2FNAS",
        "nnU-Net", "DiNTS", "Swin UNETR", "Uni. Model", "nnU-Net*"
    ]
    dsc = [51.8, 54.4, 52.8, 55.4, 58.2, 62.3, 67.2]

    # Sort by DSC
    sorted_pairs = sorted(zip(dsc, methods))
    dsc_sorted, methods_sorted = zip(*sorted_pairs)

    colors = ['lightgray'] * len(methods_sorted)
    colors[methods_sorted.index("nnU-Net*")] = [112/255, 48/255, 160/255]

    x = np.arange(len(methods_sorted))

    fig, ax = plt.subplots(figsize=args.figsize)

    bars = ax.bar(x, dsc_sorted, width=args.bar_width, color=colors,
                  edgecolor='black', linewidth=args.linewidth, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(methods_sorted, rotation=45, ha='right', fontsize=args.fontsize)
    ax.tick_params(axis='y', labelsize=args.fontsize)
    ax.set_ylabel('DSC (%)', fontsize=args.fontsize)
    ax.set_ylim(args.ylim)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=args.ytick_bins, prune=None))
    ax.tick_params(axis='both', which='major', length=args.axiswidth*8, width=args.axiswidth, pad=args.labelpad//2)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(args.linewidth)

    # Add arrow and annotation between top two bars
    best_idx = len(dsc_sorted) - 1
    second_best_idx = len(dsc_sorted) - 2
    best_val = dsc_sorted[best_idx]
    second_val = dsc_sorted[second_best_idx]
    diff = best_val - second_val

    arrow_offset = 0.25
    arrowprops = dict(arrowstyle='->', color='white', linewidth=args.linewidth)
    ax.annotate('',
                xy=(best_idx, best_val + arrow_offset),
                xytext=(best_idx, second_val + arrow_offset),
                arrowprops=arrowprops)

    ax.text(best_idx, best_val + 2.0, f"+{diff:.1f}%",
            ha='center', va='bottom', fontsize=args.fontsize, color='red', weight='bold')

    fig.tight_layout()

    out_path = Path(args.output_dir) / "fig_ePAI_won_MSD.png"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', pad_inches=0.0)
    print(f"Figure saved to {out_path}")

if __name__ == "__main__":
    args = parse_args()
    plot_performance_bar(args)
