from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from batchgenerators.utilities.file_and_folder_operations import *
import argparse
import glob # <--- This line has been added

def main(args):
    """
    Generates the dataset.json file for your specific multi-organ task,
    excluding the pancreatic lesion.
    """
    
    # This is the corrected label mapping for YOUR dataset
    labels = {
        "background": 0,
        "spleen": 1,
        "kidney_right": 2,
        "kidney_left": 3,
        "gall_bladder": 4,
        "liver": 5,
        "stomach": 6,
        "aorta": 7,
        "postcava": 8,
        "pancreas": 9,
        "adrenal_gland_right": 10,
        "adrenal_gland_left": 11,
        "lung_right": 12,
        "lung_left": 13,
        "colon": 14,
        "prostate": 15,
        "bladder": 16,
        "pancreatic_duct": 17,
        # "pancreatic_lesion" is now excluded
        "celiac_artery": 19,
        "common_bile_duct": 20,
        "duodenum": 21,
        "femur_left": 22,
        "femur_right": 23,
        "pancreas_body": 24,
        "pancreas_head": 25,
        "pancreas_tail": 26,
        "superior_mesenteric_artery": 27,
        "veins": 28
    }

    # Generate the dataset.json file using the function from nnU-Net
    generate_dataset_json(
        join(args.raw_dir, args.dataset_name),
        channel_names={0: 'CT'},
        labels=labels,
        # This line has been corrected to use glob.glob()
        num_training_cases=len(glob.glob(join(args.raw_dir, args.dataset_name, 'imagesTr', '*.nii.gz'))),
        file_ending='.nii.gz',
        dataset_name=args.dataset_name,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated the default paths to match your setup
    parser.add_argument("--raw_dir", type=str, default='/cluster/VAST/civalab/results/elham_results/ePAI_test/nnUNet_raw', help='Path to the nnU-Net raw data directory')
    parser.add_argument("--dataset_name", type=str, default='Dataset102_ePAI_Test', help='Name of the target dataset folder')
    
    args = parser.parse_args()
    main(args)
