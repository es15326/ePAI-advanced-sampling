'''
python -W ignore fig_violinplot.py
'''

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import argparse

np.random.seed(1203)  # or any fixed integer

def parse_args():
    parser = argparse.ArgumentParser(description="Violin + Boxplot for inter-annotator DSC scores")
    parser.add_argument('--csv_path', type=str, default='raw/interannotator_agreement.csv', help='Path to the CSV file')
    parser.add_argument('--output_path', type=str, default='interannotator_dsc_violin.png', help='Output PNG file name')
    parser.add_argument('--figure_width', type=float, default=7, help='Figure width in inches')
    parser.add_argument('--figure_height', type=float, default=6, help='Figure height in inches')
    parser.add_argument('--fontsize', type=int, default=26, help='Font size for all labels and ticks')
    parser.add_argument('--linewidth', type=float, default=2, help='Line width for axes and boxplot')
    parser.add_argument('--violin_color', type=float, nargs=3, default=[112/255, 48/255, 160/255],
                        help='Violin fill color as RGB (0–1 scale)')
    parser.add_argument('--violin_alpha', type=float, default=0.1, help='Transparency of violin fill color')
    parser.add_argument('--axiswidth', type=float, default=1.2, help='Width of the axes')
    parser.add_argument('--labelpad', type=int, default=10, help='Padding for labels')
    return parser.parse_args()

def main():
    args = parse_args()

    # Load and filter the data
    df = pd.read_csv(args.csv_path)
    df['tumor_dice'] = df['tumor_dice'].astype(float) * 100
    df_dsc = df[['tumor_dice']].copy()
    df_dsc = df_dsc[df_dsc['tumor_dice'] > 0.1 * 100]
    df_dsc.rename(columns={'tumor_dice': 'DSC'}, inplace=True)
    df_dsc['Structure'] = ''

    # Create the plot
    plt.figure(figsize=(args.figure_width, args.figure_height))

    # Violin plot with transparent fill
    sns.violinplot(
        data=df_dsc,
        x='Structure', y='DSC',
        inner="quart", linewidth=args.linewidth,
        color=args.violin_color,
        fill=False
    )
    plt.tick_params(axis='both', which='major', length=args.axiswidth*8, width=args.axiswidth, pad=args.labelpad//2)

    # Overlay individual points
    # Custom jittered scatter
    x_jitter = np.random.normal(loc=0, scale=0.09, size=len(df_dsc))
    for i, y in enumerate(df_dsc['DSC']):
        color = 'red' if y < 0.2 * 100 else args.violin_color
        plt.plot(x_jitter[i], y, 'o',
                markersize=8,
                color=color,
                alpha=args.violin_alpha,
                markeredgewidth=1)
        
    plt.axhline(y=0.2 * 100, color='red', linestyle='--', linewidth=args.linewidth)
    plt.text(
        x=0.25, y=(0.2 - 0.08) * 100,
        s="min. threshold",
        color='red',
        fontsize=args.fontsize,
        ha='center',
        va='bottom'
    )

    # Customize axes
    ax = plt.gca()
    ax.set_xticks([])                     # remove x ticks
    ax.set_xticklabels([])               # remove x tick labels
    ax.set_xlabel("")                    # remove x-axis label
    ax.spines['top'].set_visible(False)  # remove top frame
    ax.spines['right'].set_visible(False)  # remove right frame
    ax.spines['left'].set_linewidth(args.linewidth)
    ax.spines['bottom'].set_linewidth(args.linewidth)
    ax.tick_params(width=args.linewidth)

    # Label settings
    plt.ylabel("annotator agreement, DSC (%)", fontsize=args.fontsize)
    plt.yticks(fontsize=args.fontsize)
    plt.ylim(0, 100)
    plt.tight_layout()

    # Save
    plt.savefig(args.output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

if __name__ == "__main__":
    main()