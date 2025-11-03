'''
python -W ignore fig_ePAI_scaling.py --confAI --coi pdac,pnet,cyst --csv_files JHH-Test-PDAC\<2cm,JHH-Test-PDAC\>2cm,JHH-Test-nonPDAC\<2cm,JHH-Test-nonPDAC\>2cm,JHH-Test-Normal --gt P,P,P,P,N --experiment_name JHH-Tumor-Scaling --pct 71,100 --baseline_curve

python -W ignore fig_ePAI_scaling.py --confAI --coi pdac,pnet,cyst --csv_files JHH-Test-PDAC\<2cm,JHH-Test-PDAC\>2cm,JHH-Test-nonPDAC\<2cm,JHH-Test-nonPDAC\>2cm,JHH-Test-Normal --gt P,P,P,P,N --experiment_name JHH-PDAC 

python -W ignore fig_ePAI_scaling.py --confAI --coi pdac,pnet,cyst --csv_files JHHOUT-Test-PDAC\<2cm,JHHOUT-Test-PDAC\>2cm,JHH-Test-Normal --gt P,P,N --experiment_name JHHOUT-PDAC 

python -W ignore fig_ePAI_scaling.py --confAI --coi pdac --csv_files Heidelberg-Test-PDAC,JHHOUT-Test-PDAC\<2cm,JHHOUT-Test-PDAC\>2cm,PANORAMA-Test-PDAC\<2cm,PANORAMA-Test-PDAC\>2cm,ULS-Test-PDAC\<2cm,ULS-Test-PDAC\>2cm,UCSF-Test-PDAC\>2cm,UCSF-Test-PDAC\<2cm,UCSF-Test-Normal-AllOrgans,Heidelberg-Test-Normal,ULS-Test-Normal --gt P,P,P,P,P,P,P,P,P,N,N,N --experiment_name External-PDAC --pct 10,25,50,75,100
'''

import os
import csv
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from pprint import pprint
from tqdm import tqdm
from helper_functions import *

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Tumor Detection Results with Scaling")
    parser.add_argument('--csv_path', type=str, default='raw', help='Directory containing CSV results')
    parser.add_argument('--csv_files', type=str, default='ULS-Test-PDAC<2cm,ULS-Test-Normal', help='Comma-separated CSV file names')
    parser.add_argument('--gt', type=str, default='P,N', help='Comma-separated ground truth labels')
    parser.add_argument('--pct', type=str, default='10,25,50,75,100', help='Comma-separated training percentages')
    parser.add_argument('--confAI', action='store_true', help='Use AI confidence for filtering')
    parser.add_argument('--pancreas_volume_threshold', type=float, default=1000, help='Volume threshold in mm^3')
    parser.add_argument('--duct_volume_threshold', type=float, default=1000, help='Volume threshold in mm^3')
    parser.add_argument('--coi', type=str, default='pdac', help='Comma-separated conditions of interest')
    parser.add_argument('--diameter_threshold', type=float, default=1.0, help='Diameter threshold in mm')
    parser.add_argument('--scaling_legend', type=str, default='External test PDAC cohort', help='Legend label for scaling plot')
    parser.add_argument('--experiment_name', type=str, default='ULS-PDAC', help='Output filename prefix')
    parser.add_argument('--baseline_curve', action='store_true', help='Add baseline curve to ROC plot')
    parser.add_argument('--fontsize', type=float, default=120, help='Font size for ROC plot')
    return parser.parse_args()

def plot_roc_curve_scaling(X, GT, PR, saverocpath, roc_file_name=None,
                           axiswidth=5, linewidth=15, fontsize=120, 
                           figsize=(40, 40), operating_point=True,
                           alpha=0.0, zoomin=False, labelpad=50, 
                           legend_exp_name='', roc_color=[112/255, 48/255, 160/255],
                           baseline_curve=False,
                           ):
    plt.rcParams.update({'font.size': fontsize})
    plt.rcParams['axes.linewidth'] = axiswidth
    fig, ax = plt.subplots(figsize=figsize)

    if baseline_curve: # Optional: add a baseline curve (not organized in GT/PR format)
        df = pd.read_csv('raw/baseline_model_trained_on_MSD/roc_curve_nnunet.csv')
        auc_baseline = np.trapz(df['TPR'], df['FPR'])  # Approximate AUC using trapezoidal rule
    
        ax.plot(df['FPR'], df['TPR'], color='gray', alpha=0.3,
                label=f"MSD (AUC={auc_baseline:.3f})", 
                linewidth=linewidth)
        
        if operating_point:
            # Find the point closest to (0,1)
            distances = np.sqrt(df['FPR']**2 + (1 - df['TPR'])**2)
            op_idx = np.argmin(distances)
            op_fpr, op_tpr = df['FPR'][op_idx], df['TPR'][op_idx]
            ax.plot(op_fpr, op_tpr, 'o', markersize=3*linewidth,
                    markeredgewidth=linewidth,
                    markeredgecolor='black', markerfacecolor='white',
                    )

    for _, (pct, gt, pr) in enumerate(zip(X, GT, PR)):
        fpr, tpr, thresholds = roc_curve(gt, pr)
        auc_metric = compute_auc_metrics(gt, pr, print_auc=False)
        
        # Customize labels and colors for specific percentages
        if pct == 71:
            label = f"PANORAMA (AUC={auc_metric['auc'][0]:.3f})"
            curve_color = 'darkgray'
        elif pct == 100:
            label = f"PanTS (AUC={auc_metric['auc'][0]:.3f})"
            curve_color = roc_color
        else:
            label = f"{pct}% (AUC={auc_metric['auc'][0]:.3f})"
            curve_color = roc_color
            
        ax.plot(fpr, tpr, label=label, linewidth=linewidth, color=curve_color)

        if operating_point:
            # Find the point closest to (0,1)
            distances = np.sqrt(fpr**2 + (1 - tpr)**2)
            op_idx = np.argmin(distances)
            op_fpr, op_tpr = fpr[op_idx], tpr[op_idx]
            op_threshold = thresholds[op_idx]
            ax.plot(op_fpr, op_tpr, 'o', markersize=3*linewidth,
                    markeredgewidth=linewidth,
                    markeredgecolor='black', markerfacecolor='white',
                    )
    
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0., 1.02)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel('1-Specificity', labelpad=labelpad)
    ax.set_ylabel('Sensitivity', labelpad=labelpad)
    ax.tick_params(axis='both', which='major', length=axiswidth*8, width=axiswidth, pad=labelpad//2)
    ax.grid(alpha=alpha, linewidth=linewidth)
    os.makedirs(saverocpath, exist_ok=True)
    fig.savefig(os.path.join(saverocpath, f'{roc_file_name}.png'), bbox_inches='tight', pad_inches=0.0, dpi=200)
    print(f"ROC curve saved to {os.path.join(saverocpath, f'{roc_file_name}.png')}")

def event(pid_result, gt, args):
    pid = pid_result['bdmap_id']
    conf = get_AI_confidence(pid_result, args)
    pr = get_AI_prediction(pid_result, args)
    logit = get_AI_logit(pid_result, args)

    with open(args.internal_csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([pid, pr, logit, conf, gt])

def evaluate_scaling_auc(args):
    auc = []
    processed_gt_results = []
    processed_logit_results = []
    for pct in tqdm(args.pct, total=len(args.pct), ncols=60):
        args.internal_csv_path = initialize_temporary_csv()
        csv_paths = [os.path.join(args.csv_path, f"{pct}pct", f"{name}.csv") for name in args.csv_files]

        pid_results, gt_results = read_standardized_csv(args.gt, csv_paths)
        
        for pid_result, gt in zip(pid_results, gt_results):
            event(pid_result, gt, args)

        processed = read_temporary_csv(args, remove_csv=True)
        
        if args.confAI:
            processed = [x for x in processed if int(x['confidence']) >= 0.5]

        gt=[int(x['gt']) for x in processed]
        logit=[float(x['logit']) for x in processed]

        auc_metric = compute_auc_metrics(
            gt=gt,
            logit=logit,
            print_auc=False
        )
        
        auc.append(auc_metric)
        processed_gt_results.append(gt)
        processed_logit_results.append(logit)
        
    x_values = [int(p) for p in args.pct]
    label = f"{args.scaling_legend} (n={len(gt_results)})"
    plot_scaling_result(auc, 
                        legend_exp_name=label, 
                        file_name=args.experiment_name, 
                        x_values=x_values,
                        )
    plot_roc_curve_scaling(x_values,
                           processed_gt_results, processed_logit_results,
                           saverocpath='roc',
                           roc_file_name=args.experiment_name,
                           legend_exp_name=label,
                           axiswidth=12, linewidth=25, fontsize=args.fontsize, 
                           figsize=(40, 40), operating_point=True,
                           alpha=0.0, zoomin=False,
                           baseline_curve=args.baseline_curve,
                           )

def main():
    args = parse_args()
    args.pct = [p.strip() for p in args.pct.split(',')]
    args.gt = [g.strip() for g in args.gt.split(',')]
    args.csv_files = [f.strip() for f in args.csv_files.split(',')]
    args.coi = [c.strip() for c in args.coi.split(',')]

    print("\nParsed Arguments:")
    pprint(vars(args), sort_dicts=False)

    if not os.path.exists(args.csv_path):
        raise FileNotFoundError(f"CSV path {args.csv_path} not found.")

    evaluate_scaling_auc(args)

if __name__ == "__main__":
    main()