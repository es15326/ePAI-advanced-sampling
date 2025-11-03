import glob, os

path = '/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task1007-YC_3mm_6cls_Remap_dataset/labelsTs/'
# path = '/mnt/ccvl15/yucheng/KidneyDiff/3D-TransUNet/out/Task1007-YC_3mm_6cls_Remap_dataset/GeTU901_epo250/NoPNET/'

tar_path = '/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task1007-YC_3mm_6cls_Remap_dataset/labelsTs_3cls/'
# tar_path = '/mnt/ccvl15/yucheng/KidneyDiff/3D-TransUNet/out/Task1007-YC_3mm_6cls_Remap_dataset/GeTU901_epo250/3cls/'

os.makedirs(tar_path, exist_ok=True)
case_list = glob.glob(path+'*.nii.gz')

import nibabel as nib
import numpy as np
from tqdm import tqdm

# 1: pancreas
# 2: duct
# 3: pdac
# 4: cyst
# 5: pnet

for case in tqdm(case_list):
    data = nib.load(case)
    img = data.get_fdata()
    affine = data.affine

    new_img = np.zeros_like(img)
    new_img[img==1] = 1
    new_img[img==2] = 1
    new_img[img==3] = 2
    new_img[img==4] = 2
    new_img[img==5] = 2
    # print(np.unique(img))
    new_data = nib.Nifti1Image(new_img, affine)
    # print(case.replace('pred', 'pred-merge3cls'))
    nib.save(new_data, tar_path+case.split('/')[-1])
