'''
python -W ignore fig_ePAI_multiclass.py --csv_2cls raw/2cls_DSC.csv --csv_28cls raw/28cls_DSC.csv --output fig_multiclass_comparison.png
'''

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_rel
from statannotations.Annotator import Annotator

def parse_args():
    parser = argparse.ArgumentParser(description="Compare tumor Dice scores using violin plot and stripplot")
    parser.add_argument('--csv_2cls', type=str, required=True, help='Path to 2-class DSC CSV')
    parser.add_argument('--csv_28cls', type=str, required=True, help='Path to 28-class DSC CSV')
    parser.add_argument('--output', type=str, default='tumor_dice_comparison_violin.png', help='Output plot filename')
    parser.add_argument('--figsize', type=float, nargs=2, default=[6, 7], help='Figure size (width height)')
    parser.add_argument('--fontsize', type=int, default=20, help='Font size for labels and ticks')
    parser.add_argument('--linewidth', type=float, default=1.8, help='Line width for plot elements')
    parser.add_argument('--highlight_color', type=float, nargs=3, default=[112/255, 48/255, 160/255], help='RGB color for 28cls')
    parser.add_argument('--axiswidth', type=float, default=1.9, help='Width of the axes')
    parser.add_argument('--labelpad', type=int, default=10, help='Padding for labels')
    return parser.parse_args()

def main(args):
    # Load and merge data
    df_2cls = pd.read_csv(args.csv_2cls)[['case_name', 'tumor_dice']] * 100
    df_28cls = pd.read_csv(args.csv_28cls)[['case_name', 'tumor_dice']] * 100
    merged = pd.merge(df_2cls, df_28cls, on='case_name', suffixes=('_2cls', '_28cls'))

    # Filter non-zero tumor dice
    filtered = merged[(merged['tumor_dice_2cls'] > 0) & (merged['tumor_dice_28cls'] > 0)]

    # Paired t-test
    t_stat, p_value = ttest_rel(filtered['tumor_dice_2cls'], filtered['tumor_dice_28cls'])

    # Prepare for plotting
    df_long = pd.melt(filtered, id_vars='case_name',
                      value_vars=['tumor_dice_2cls', 'tumor_dice_28cls'],
                      var_name='Method', value_name='Tumor_DSC')
    df_long['Method'] = df_long['Method'].map({
        'tumor_dice_2cls': '2cls',
        'tumor_dice_28cls': '28cls'
    })

    # Plot
    plt.figure(figsize=tuple(args.figsize))
    palette = {'2cls': 'gray', '28cls': tuple(args.highlight_color)}
    ax = sns.violinplot(x='Method', y='Tumor_DSC', data=df_long, palette=palette, 
                        linewidth=args.linewidth, inner="quart", cut=0, fill=False)
    sns.stripplot(x='Method', y='Tumor_DSC', data=df_long, palette=palette,
                  jitter=False, marker='o', alpha=0.1,
                  size=5 * args.linewidth, edgecolor='lightgray', linewidth=args.linewidth)

    # Statistical annotation
    pairs = [('2cls', '28cls')]
    annotator = Annotator(ax, pairs, data=df_long, x='Method', y='Tumor_DSC')
    annotator.configure(
        test='t-test_paired',
        text_format='star',
        loc='outside',
        verbose=False,
        fontsize=args.fontsize,
    )
    annotator.apply_and_annotate()

    # mean_diff = filtered['tumor_dice_28cls'].mean() - filtered['tumor_dice_2cls'].mean()
    # plt.text(x=0.5, y=0.99*100, s=f"Δmean = {mean_diff:.1f}", ha='center', fontsize=args.fontsize)

    # Style
    ax.set_ylabel("tumor segmentation, DSC (%)", fontsize=args.fontsize)
    ax.set_ylim(0.4 * 100, 1.0 * 100)
    ax.set_yticks([40, 60, 80, 100])
    ax.set_xlabel("")
    ax.set_xticklabels(["2 classes", "28 classes"], fontsize=args.fontsize)
    ax.tick_params(labelsize=args.fontsize, width=args.linewidth)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(args.linewidth)
    ax.spines['bottom'].set_linewidth(args.linewidth)

    plt.tick_params(axis='both', which='major', length=args.axiswidth*5, width=args.axiswidth, pad=args.labelpad//2)

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight', pad_inches=0.0)
    print(f"Saved violin plot to {args.output} (n={len(filtered)} matched cases).")

if __name__ == "__main__":
    args = parse_args()
    main(args)