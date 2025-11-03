import json, glob, os


datast = "Task1009-YC_3mm_6cls_Syn_v1_dataset"

template = {
    "name": datast,
    "modality": {
        "0": "ct"
    },
    "labels": {
        "0": "background",
        "1": "pancreas",
        "2": "duct",
        "3": "pdac",
        "4": "cyst",
        "5": "pnet"
    },
    "numTraining": 0,
    "numTest": 0,
    "training": [],
    "test": []
}

root = '/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task1007-YC_3mm_6cls_Remap_dataset/Syn_v1/'
image_folders = ['imagesTr_pdac','imagesTr_cyst', 'imagesTr_pnet', 'imagesTr_healthy']
label_folders = ['labelsTr_pdac','labelsTr_cyst', 'labelsTr_pnet', 'labelsTr_healthy']

train_list = glob.glob(os.path.join(root, 'imagesTr_pdac/*')) + glob.glob(os.path.join(root, 'imagesTr_cyst/*')) + glob.glob(os.path.join(root, 'imagesTr_pnet/*')) + glob.glob(os.path.join(root, 'imagesTr_healthy/*'))




train_list.sort()

for i in train_list:
    template['training'].append({'image': './imagesTr/'+i.split('/')[-1].replace('_0000', ''), 'label': './labelsTr/'+i.split('/')[-1].replace('_0000', '')})



template['numTraining'] = len(train_list)

with open('dataset.json', 'w') as f:
    json.dump(template, f)


#####


target_path = '/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/'+datast

import os, shutil
from tqdm import tqdm
os.makedirs(os.path.join(target_path, 'labelsTr'), exist_ok=True)
os.makedirs(os.path.join(target_path, 'imagesTr'), exist_ok=True)

for train in tqdm(train_list):
    shutil.copy(train, os.path.join(target_path, 'imagesTr'))
    shutil.copy(train.replace('imagesTr', 'labelsTr').replace('_0000', ''), os.path.join(target_path, 'labelsTr', train.split('/')[-1].replace('_0000', '')))

