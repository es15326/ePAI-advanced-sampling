'''
python -W ignore fig_ePAI_multiclass_barplot.py --file_2cls raw/2cls_DSC.csv --file_28cls raw/28cls_DSC.csv --save_path fig_multiclass_comparison_barplot.png
'''

import argparse
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

def load_and_prepare_data(file_2cls, file_28cls):
    df_2cls = pd.read_csv(file_2cls)
    df_28cls = pd.read_csv(file_28cls)

    # Match on common case names
    common_cases = set(df_2cls['case_name']).intersection(df_28cls['case_name'])
    df_2cls = df_2cls[df_2cls['case_name'].isin(common_cases)].set_index('case_name')
    df_28cls = df_28cls[df_28cls['case_name'].isin(common_cases)].set_index('case_name')

    # Extract tumor DSC
    dsc_2cls = df_2cls['tumor_dice'] * 100  # Convert to percent
    dsc_28cls = df_28cls['tumor_dice'] * 100

    return dsc_2cls, dsc_28cls

def plot_comparison(dsc_2cls, dsc_28cls, font_size, fig_size, linewidth, save_path=None):
    
    mpl.rcParams.update({'font.size': font_size})

    def bootstrap_std(series, sample_frac=0.95, num_samples=10):
        means = []
        n = int(len(series) * sample_frac)
        for _ in range(num_samples):
            sample = series.sample(n=n, replace=False)
            means.append(sample.mean())
        return np.std(means)

    data = [dsc_2cls.mean(), dsc_28cls.mean()]
    errors = [
        bootstrap_std(dsc_2cls),
        bootstrap_std(dsc_28cls)
    ]
    labels = ['2-class\nmodel', '28-class\nmodel']
    colors = ['lightgray', (112/255, 48/255, 160/255)]

    fig, ax = plt.subplots(figsize=fig_size)
    bars = ax.bar(
        labels, data, yerr=errors, capsize=15,
        color=colors, edgecolor='black', linewidth=linewidth, width=0.5,
        error_kw=dict(linewidth=linewidth, capthick=linewidth)
    )

    ax.set_ylabel('tumor segmentation, DSC (%)')
    ax.set_ylim(50, 70)
    ax.set_yticks([50, 55, 60, 65, 70])  # Adjust as needed
    # right after plotting your bars, e.g. just before saving/showing:
    ax.margins(x=0.2)   # 20% padding on each side

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['bottom'].set_linewidth(linewidth)

    plt.tick_params(axis='both', which='major', length=8 * linewidth, width=linewidth, pad=4*linewidth)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0.0)
    else:
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="Compare 2-class and 28-class DSC scores with bar plot.")
    parser.add_argument('--file_2cls', type=str, required=True, help='Path to 2-class DSC CSV file')
    parser.add_argument('--file_28cls', type=str, required=True, help='Path to 28-class DSC CSV file')
    parser.add_argument('--font_size', type=int, default=22, help='Font size for plot text')
    parser.add_argument('--fig_width', type=float, default=3.8, help='Figure width in inches')
    parser.add_argument('--fig_height', type=float, default=9.0, help='Figure height in inches')
    parser.add_argument('--linewidth', type=float, default=2.5, help='Bar edge linewidth')
    parser.add_argument('--save_path', type=str, default=None, help='Optional path to save the figure')

    args = parser.parse_args()

    dsc_2cls, dsc_28cls = load_and_prepare_data(args.file_2cls, args.file_28cls)
    plot_comparison(
        dsc_2cls, dsc_28cls,
        font_size=args.font_size,
        fig_size=(args.fig_width, args.fig_height),
        linewidth=args.linewidth,
        save_path=args.save_path
    )

if __name__ == "__main__":
    main()