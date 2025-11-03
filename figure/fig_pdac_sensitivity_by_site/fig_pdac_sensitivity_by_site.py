'''
python -W ignore fig_pdac_sensitivity_by_site.py
'''

import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
from matplotlib.ticker import MaxNLocator

def parse_args():
    parser = argparse.ArgumentParser(description="Plot PDAC detection sensitivity across sites")
    parser.add_argument('--figsize', type=float, nargs=2, default=[16, 8], help='Figure size (width height)')
    parser.add_argument('--fontsize', type=int, default=25, help='Font size for labels and ticks')
    parser.add_argument('--linewidth', type=float, default=1, help='Line width for bars and spines')
    parser.add_argument('--bar_width', type=float, default=0.35, help='Width of each bar')
    parser.add_argument('--ytick_bins', type=int, default=4, help='Number of bins in the y-axis')
    parser.add_argument('--ylim', type=float, nargs=2, default=[0, 100], help='Y-axis limit')
    parser.add_argument('--offset_y', type=float, default=50, help='Offset for text labels on the y-axis')
    parser.add_argument('--xtick_rotation', type=float, default=45, help='Rotation angle for x-axis tick labels')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save the output figure')
    return parser.parse_args()

def plot_sensitivity_figure(args):
    sites = ['JHH\n(43|236)', # JHH
             'A\n(48|318)', # JHH-OUT
             'B\n(80|120)', # MSD-Pancreas
             'C\n(97|102)', # ULS
             'D\n(130|887)', # UCSF 
             'E\n(206|362)', # PANORAMA
             'F\n(110|411)', # CoH
            ]
    internal_sites = ['JHH\n(43|236)']
    external_sites = [s for s in sites if s not in internal_sites]

    sensitivity_lt2cm = [95.3, 
                         97.9, 
                         100, 
                         90.7, 
                         89.2, 
                         87.9, 
                         92.7,
                         ]
    sensitivity_gt2cm = [97.5, 
                         98.1, 
                         99.2, 
                         100, 
                         97.1, 
                         97.2, 
                         98.3,
                         ]

    color_lt2cm = [237/255, 16/255, 105/255]
    color_gt2cm = 'lightgray'

    x = np.arange(len(sites))
    fig, ax = plt.subplots(figsize=args.figsize)

    ax.bar(x - args.bar_width/2, sensitivity_lt2cm, args.bar_width, label='PDAC (≤2 cm)',
           color=color_lt2cm, edgecolor='black', linewidth=args.linewidth)
    ax.bar(x + args.bar_width/2, sensitivity_gt2cm, args.bar_width, label='PDAC (>2 cm)',
           color=color_gt2cm, edgecolor='black', linewidth=args.linewidth)

    ax.set_ylabel('Proportion of PDACs\ndetected by ePAI (%)', fontsize=args.fontsize)
    ax.set_xticks(x)
    ax.set_xticklabels(sites, fontsize=args.fontsize, rotation=args.xtick_rotation)
    ax.tick_params(axis='x', which='major', pad=0, labelbottom=True)
    ax.set_ylim(args.ylim)
    ax.tick_params(axis='y', labelsize=args.fontsize, width=args.linewidth)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=args.ytick_bins, prune=None))

    legend = ax.legend(loc='lower right', fontsize=args.fontsize, framealpha=1.0, frameon=True, facecolor='white')
    legend.get_frame().set_edgecolor('black')
    legend.get_frame().set_linewidth(args.linewidth)

    if internal_sites:
        internal_idx = [sites.index(site) for site in internal_sites]
        ax.text(np.mean(internal_idx), args.ylim[0] - args.offset_y, 'internal', ha='center', va='center',
                fontsize=args.fontsize, transform=ax.transData)

    if external_sites:
        external_idx = [sites.index(site) for site in external_sites]
        ax.text(np.mean(external_idx), args.ylim[0] - args.offset_y, 'external', ha='center', va='center',
                fontsize=args.fontsize, transform=ax.transData)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(args.linewidth)

    plt.tight_layout()
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, 'fig_pdac_sensitivity_by_site.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {output_path}")

if __name__ == '__main__':
    args = parse_args()
    plot_sensitivity_figure(args)
