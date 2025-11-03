
import os, shutil

image_root = '/data/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task1007-YC_3mm_6cls_Remap_dataset/imagesTs/'

# label_root = '/data/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task1006-YC_3mm_6cls_Padded_dataset/labelsTs_3cls/'
label_root = '/data/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task1007-YC_3mm_6cls_Remap_dataset/labelsTs/'

# pred_root = '/mnt/ccvl15/yucheng/KidneyDiff/3D-TransUNet/out/Task1004-YC_3mm_6cls_Padded_TumorConnected_dataset/GeTU901_epo250/3cls/'
pred_root = '/mnt/ccvl15/yucheng/KidneyDiff/3D-TransUNet/out/Task1008-YC_3mm_6cls_Remap_Test_dataset/GeTU1008_3DTransUNet_epo500/all/imagesTs-onehot/'

error_cases = [
'FELIX5627_ARTERIAL.nii.gz',
'FELIX7382_VENOUS.nii.gz',
'FELIX-PDAC-1074_ARTERIAL.nii.gz',
'FELIX-PDAC-1074_ARTERIAL.nii.gz',
'FELIX7527_VENOUS.nii.gz',
'FELIX5285_ARTERIAL.nii.gz',
'FELIX5285_ARTERIAL.nii.gz',
'FELIX-PDAC-1054_ARTERIAL.nii.gz',
'FELIX5536_VENOUS.nii.gz',
'FELIX5589_ARTERIAL.nii.gz',
'FELIX5589_ARTERIAL.nii.gz',
'FELIX7394_ARTERIAL.nii.gz',
'FELIX7394_ARTERIAL.nii.gz',
'FELIX-PDAC-1148_VENOUS.nii.gz',
'FELIX7382_ARTERIAL.nii.gz',
'FELIX7382_ARTERIAL.nii.gz',
'FELIX7382_ARTERIAL.nii.gz',
'FELIX7382_ARTERIAL.nii.gz',
'FELIX-PDAC-1071_VENOUS.nii.gz',
'FELIX5370_VENOUS.nii.gz',
'FELIX5377_VENOUS.nii.gz',
'FELIX5377_VENOUS.nii.gz',
'FELIX-CYS-1207_VENOUS.nii.gz',
'FELIX-PDAC-1070_ARTERIAL.nii.gz',
]


# for error_case in error_cases:
#     if os.path.exists(image_root+error_case+"_ARTERIAL.nii.gz"):
#         os.makedirs('./review_cases/{}/ARTERIAL/'.format(error_case), exist_ok=True)
#         shutil.copy(image_root+error_case+"_ARTERIAL.nii.gz", './review_cases/{}/ARTERIAL/'.format(error_case)+'ct.nii.gz')

#     if os.path.exists(image_root+error_case+"_VENOUS.nii.gz"):
#         os.makedirs('./review_cases/{}/VENOUS/'.format(error_case), exist_ok=True)
#         shutil.copy(image_root+error_case+"_VENOUS.nii.gz", './review_cases/{}/VENOUS/'.format(error_case)+'ct.nii.gz')

# for error_case in error_cases:
#     if os.path.exists(label_root+error_case+"_ARTERIAL.nii.gz"):
#         os.makedirs('./review_cases/{}/ARTERIAL/'.format(error_case), exist_ok=True)
#         shutil.copy(label_root+error_case+"_ARTERIAL.nii.gz", './review_cases/{}/ARTERIAL/'.format(error_case)+'label.nii.gz')

#     if os.path.exists(label_root+error_case+"_VENOUS.nii.gz"):
#         os.makedirs('./review_cases/{}/VENOUS/'.format(error_case), exist_ok=True)
#         shutil.copy(label_root+error_case+"_VENOUS.nii.gz", './review_cases/{}/VENOUS/'.format(error_case)+'label.nii.gz')

# for error_case in error_cases:
#     if os.path.exists(pred_root+error_case+"_ARTERIAL.nii.gz"):
#         os.makedirs('./review_cases/{}/ARTERIAL/'.format(error_case), exist_ok=True)
#         shutil.copy(pred_root+error_case+"_ARTERIAL.nii.gz", './review_cases/{}/ARTERIAL/'.format(error_case)+'prediction.nii.gz')

#     if os.path.exists(pred_root+error_case+"_VENOUS.nii.gz"):
#         os.makedirs('./review_cases/{}/VENOUS/'.format(error_case), exist_ok=True)
#         shutil.copy(pred_root+error_case+"_VENOUS.nii.gz", './review_cases/{}/VENOUS/'.format(error_case)+'prediction.nii.gz')