'''
python -W ignore fig_dataset_characteristics.py 
'''

import matplotlib.pyplot as plt
import numpy as np
import argparse
from pathlib import Path
from matplotlib.ticker import MaxNLocator

def parse_args():
    parser = argparse.ArgumentParser(description="Plot broken y-axis bar chart for pancreas datasets")
    parser.add_argument('--figsize', type=float, nargs=2, default=[5, 6], help='Figure size (width height)')
    parser.add_argument('--fontsize', type=int, default=14, help='Font size for labels and ticks')
    parser.add_argument('--linewidth', type=float, default=1, help='Line width for bars and spines')
    parser.add_argument('--bar_width', type=float, default=0.5, help='Width of each bar')
    parser.add_argument('--ytick_bins', type=int, default=5, help='Number of bins in the y-axis (bottom plot)')
    parser.add_argument('--ytick_bins_top', type=int, default=2, help='Number of bins in the y-axis (top plot)')
    parser.add_argument('--xtick_rotation', type=float, default=45, help='Rotation angle for x-axis tick labels')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output figure')
    return parser.parse_args()

def plot_broken_axis_bar(args):
    datasets = [
        "CTpred-Sunitinib", "Pancreatic-CT-CBCT-SEG",
        "CPTAC-PDA", "PCL", "MSD Pancreas", "PANORAMA", "PanTS"
    ]
    cases = np.array([38, 40, 168, 221, 420, 3000, 36390])
    x = np.arange(len(datasets))

    colors = ['lightgray'] * len(datasets)
    colors[-1] = [112/255, 48/255, 160/255]  # PanTS highlighted

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, sharex=True, figsize=args.figsize,
        gridspec_kw={"height_ratios": [1, 3]}
    )

    # Bars on both axes
    ax_top.bar(x, cases, width=args.bar_width, color=colors, edgecolor='black', linewidth=args.linewidth)
    ax_bottom.bar(x, cases, width=args.bar_width, color=colors, edgecolor='black', linewidth=args.linewidth)

    # Y-axis limits for break between 4000 and 10000, overall from 0 to 30000
    ax_top.set_ylim(10000, 45000)
    ax_bottom.set_ylim(0, 4000)

    # Apply major locator to both top and bottom axes
    ax_top.yaxis.set_major_locator(MaxNLocator(nbins=args.ytick_bins_top, prune=None))
    ax_bottom.yaxis.set_major_locator(MaxNLocator(nbins=args.ytick_bins, prune=None))

    # Hide spines where break occurs and remove unnecessary borders
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.spines['left'].set_linewidth(args.linewidth)

    ax_bottom.spines['top'].set_visible(False)
    ax_bottom.spines['right'].set_visible(False)
    ax_bottom.spines['left'].set_linewidth(args.linewidth)
    ax_bottom.spines['bottom'].set_linewidth(args.linewidth)

    # Diagonal break marks
    d = 0.01
    kwargs = dict(color="k", clip_on=False, linewidth=args.linewidth)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kwargs)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kwargs)

    # Labels and ticks
    fig.text(0.00, 0.65, "number of patients", va='center', rotation='vertical', fontsize=args.fontsize)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(datasets, fontsize=args.fontsize, rotation=args.xtick_rotation, ha="right")
    ax_top.tick_params(labeltop=False, length=4, labelsize=args.fontsize, axis='y')
    ax_top.tick_params(axis='x', length=0)
    ax_bottom.tick_params(labelsize=args.fontsize)

    fig.tight_layout()

    out_png = Path(args.output_dir) / "pancreatic_dataset_size.png"
    fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.0)
    print(f"Figure saved to {out_png}")

if __name__ == "__main__":
    args = parse_args()
    plot_broken_axis_bar(args)
