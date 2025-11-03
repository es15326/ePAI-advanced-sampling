import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re

# Example data for demonstration (replace with your actual data)
np.random.seed(42)
train_age = np.random.normal(50, 15, 1000)  # Example data for Train
val_age = np.random.normal(55, 15, 800)    # Example data for Validation

# read data from csv
def read_diameter_from_csv(file_path):
    
    data = pd.read_csv(file_path)
    data = data['lesion_radius (mm)'] * 2.0
    data = [x for x in data if x != 0]
    
    return data

def read_diameter_from_pro_csv(file_path):
    
    data = pd.read_csv(file_path)
    data = data['largest_pancreatic_tumor_diameter']
    data = [x for x in data if x != 0]
    
    return data

def read_case_shape_from_csv(file_path):
    
    data = pd.read_csv(file_path)
    data = data['case_shape']
    data = [list(map(int, re.findall(r'\d+', x))) for x in data]
    for i in range(len(data)):
        if data[i][0] == data[i][1]:
            data[i] = data[i][-1]
        else:
            data[i] = data[i][0]
    
    return data

def read_case_shape_from_pro_csv(file_path):

    data = pd.read_csv(file_path)

    # Filter out rows where case_name contains "BDMAP_A"
    data = data[~data['BDMAP ID'].str.contains("BDMAP_A", na=False)]

    data = data['case_shape']
    data = [list(map(int, re.findall(r'\d+', x))) for x in data]
    for i in range(len(data)):
        if data[i][0] == data[i][1]:
            data[i] = data[i][-1]
        else:
            data[i] = data[i][0]

    return data

def read_z_spacing_from_csv(file_path):
    
    data = pd.read_csv(file_path)
    data = data['case_spacing (mm)']
    data = [list(map(float, re.findall(r'\d+\.\d+', x))) for x in data]
    for i in range(len(data)):
        if data[i][0] == data[i][1]:
            data[i] = data[i][-1]
        else:
            data[i] = data[i][0]
    
    return data

def read_z_spacing_from_pro_csv(file_path):

    data = pd.read_csv(file_path)

    # Filter out rows where case_name contains "BDMAP_A"
    data = data[~data['BDMAP ID'].str.contains("BDMAP_A", na=False)]

    data = data['case_spacing (mm)']
    data = [list(map(float, re.findall(r'[0-9.]+', x))) for x in data]
    for i in range(len(data)):
        if data[i][0] == data[i][1]:
            data[i] = data[i][-1]
        else:
            data[i] = data[i][0]

    return data

def read_xy_spacing_from_csv(file_path):
    
    data = pd.read_csv(file_path)
    data = data['case_spacing (mm)']
    data = [list(map(float, re.findall(r'\d+\.\d+', x))) for x in data]
    for i in range(len(data)):
        if data[i][0] == data[i][1]:
            data[i] = data[i][0]
        else:
            data[i] = data[i][-1]
    
    return data

def read_xy_spacing_from_pro_csv(file_path):

    data = pd.read_csv(file_path)

    # Filter out rows where case_name contains "BDMAP_A"
    data = data[~data['BDMAP ID'].str.contains("BDMAP_A", na=False)]
    
    data = data['case_spacing (mm)']
    data = [list(map(float, re.findall(r'[0-9.]+', x))) for x in data]
    for i in range(len(data)):
        if data[i][0] == data[i][1]:
            data[i] = data[i][0]
        else:
            data[i] = data[i][-1]

    return data

jhh_train_lesion_diameter = read_diameter_from_csv('metadata/jhh_train.csv')
jhh_test_lesion_diameter = read_diameter_from_csv('metadata/jhh_test.csv')
msd_train_lesion_diameter = read_diameter_from_csv('metadata/msd_train.csv')
msd_test_lesion_diameter = read_diameter_from_csv('metadata/msd_test.csv')
panorama_lesion_diameter = read_diameter_from_csv('metadata/panorama.csv')

jhh_train_case_shape = read_case_shape_from_csv('metadata/jhh_train.csv')
jhh_test_case_shape = read_case_shape_from_csv('metadata/jhh_test.csv')
msd_train_case_shape = read_case_shape_from_csv('metadata/msd_train.csv')
msd_test_case_shape = read_case_shape_from_csv('metadata/msd_test.csv')
panorama_case_shape = read_case_shape_from_csv('metadata/panorama.csv')

jhh_train_z_spacing = read_z_spacing_from_csv('metadata/jhh_train.csv')
jhh_test_z_spacing = read_z_spacing_from_csv('metadata/jhh_test.csv')
msd_train_z_spacing = read_z_spacing_from_csv('metadata/msd_train.csv')
msd_test_z_spacing = read_z_spacing_from_csv('metadata/msd_test.csv')
panorama_z_spacing = read_z_spacing_from_csv('metadata/panorama.csv')

jhh_train_xy_spacing = read_xy_spacing_from_csv('metadata/jhh_train.csv')
jhh_test_xy_spacing = read_xy_spacing_from_csv('metadata/jhh_test.csv')
msd_train_xy_spacing = read_xy_spacing_from_csv('metadata/msd_train.csv')
msd_test_xy_spacing = read_xy_spacing_from_csv('metadata/msd_test.csv')
panorama_xy_spacing = read_xy_spacing_from_csv('metadata/panorama.csv')

# pro_lesion_diameter = read_diameter_from_pro_csv('metadata/pro.csv')
pro_xy_spacing = read_xy_spacing_from_pro_csv('metadata/pro.csv')
pro_z_spacing = read_z_spacing_from_pro_csv('metadata/pro.csv')
pro_case_shape = read_case_shape_from_csv('metadata/pro.csv')

def metadata_analysis_kdeplot(data, color, label, 
                              metadata='', figsize=(32, 12), dpi=500,
                              linewidth=10, fontsize=60, alpha=0.0,
                              xlim_min=0, xlim_max=60,
                              legend=True,
                              ):

    # Create the figure
    plt.figure(figsize=figsize, dpi=dpi)

    # Plot the Validation distribution without filled color
    for i in range(len(data)):
        sns.kdeplot(data[i], color=color[i], label=label[i], fill=True, alpha=alpha, linewidth=linewidth)

    # Add title and labels
    # plt.title("Age distribution", fontsize=14, weight="bold")
    plt.xlabel(metadata, fontsize=fontsize)
    plt.ylabel("")
    plt.yticks([])

    # Customize legend
    if legend:
        plt.legend(title="", loc="upper right", fontsize=fontsize, frameon=False)

    # Add grid lines (optional)
    plt.grid(False)

    # Set x and y axis limits if needed (adjust to match your data range)
    plt.xlim(xlim_min, xlim_max)

    # Customize x-axis tick font size
    plt.xticks(fontsize=fontsize)

    # Customize the plot: Remove upper and right spines
    ax = plt.gca()  # Get current axis
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_linewidth(linewidth*0.5)

    # Save or display the figure
    plt.tight_layout()
    if metadata != '':
        plt.savefig(os.path.join(metadata+".png"), bbox_inches="tight", pad_inches=0)
    # plt.show()

# print(len(pro_xy_spacing))
# metadata_analysis_kdeplot(data=[pro_z_spacing], 
#                           color=['#3333FF'], 
#                           label=['Pro'],
#                           metadata='Pro Z spacing (mm)',
#                           xlim_min=0, xlim_max=10,
#                           legend=False, alpha=0.5,
#                           )
# metadata_analysis_kdeplot(data=[pro_xy_spacing],
#                             color=['#3333FF'],
#                             label=['Pro'],
#                             metadata='Pro XY spacing (mm)',
#                             xlim_min=0, xlim_max=4,
#                             legend=False, alpha=0.5,
#                             )
# metadata_analysis_kdeplot(data=[pro_case_shape],
#                             color=['#3333FF'],
#                             label=['Pro'],
#                             metadata='Pro number of CT slices',
#                             xlim_min=0, xlim_max=1500,
#                             legend=False, alpha=0.5,
#                             )

# metadata_analysis_kdeplot(data=[pro_lesion_diameter, msd_train_lesion_diameter, msd_test_lesion_diameter, panorama_lesion_diameter, jhh_test_lesion_diameter], 
#                           color=['black', '#3333FF', '#8080FF', 'orange', '#ED1069'], 
#                           label=['Train (25K)', 'Train (MSD)', 'Internal Test (MSD)', 'External Test (PANORAMA)', 'External Test (Private Data)'],
#                           metadata='tumor diameter (mm)',
#                           xlim_min=0, xlim_max=80,
#                           legend=False, alpha=0.05,
#                           )
metadata_analysis_kdeplot(data=[pro_case_shape, msd_train_case_shape, msd_test_case_shape, panorama_case_shape, jhh_test_case_shape], 
                          color=['black', '#3333FF', '#8080FF', 'orange', '#ED1069'], 
                          label=['Train (25K)', 'Train (MSD)', 'Internal Test (MSD)', 'External Test (PANORAMA)', 'External Test (Private Data)'],
                          metadata='number of CT slices',
                          xlim_min=0, xlim_max=1200, 
                          legend=False, alpha=0.05,
                          )
metadata_analysis_kdeplot(data=[pro_z_spacing, msd_train_z_spacing, msd_test_z_spacing, panorama_z_spacing],
                          color=['black', '#3333FF', '#8080FF', 'orange', '#ED1069'],
                          label=['Train (25K)', 'Train (MSD)', 'Internal Test (MSD)', 'External Test (PANORAMA)'],
                          metadata='slice thickness (mm)',
                          xlim_min=0, xlim_max=10,
                          legend=False, alpha=0.05,
                          )
metadata_analysis_kdeplot(data=[pro_xy_spacing, msd_train_xy_spacing, msd_test_xy_spacing, panorama_xy_spacing, jhh_test_xy_spacing],
                          color=['black', '#3333FF', '#8080FF', 'orange', '#ED1069'],
                          label=['Train (25K)', 'Train (MSD)', 'Internal Test (MSD)', 'External Test (PANORAMA)', 'External Test (Private Data)'],
                          metadata='in-plane spacing (mm)',
                          xlim_min=0.2, xlim_max=3.2,
                          legend=False, alpha=0.05,
                          )