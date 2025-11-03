import json
template = {
    "name": "Task1008-YC_0.5mm_6cls_Remap_dataset",
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

train = '/mnt/ccvl15/yucheng/KidneyDiff/DiffTumor/STEP3.SegmentationModel/utils_data/4.felix_train_padded.list'
test = '/mnt/ccvl15/yucheng/KidneyDiff/DiffTumor/STEP3.SegmentationModel/utils_data/4.felix_test_padded.list'
# OSError: [Errno 28] No space left on device: '/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/KidneyDiff_Dataset_Cleaned/0.5mm/img/FELIX-PDAC-1002_VENOUS.nii.gz' -> '/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task1008-YC_0.5mm_6cls_Remap_dataset/imagesTs/FELIX-PDAC-1002_VENOUS.nii.gz'
train_list = open(train, 'r').readlines()
test_list = open(test, 'r').readlines()

train_list = [i.strip() for i in train_list]
train_list.sort()
test_list = [i.strip() for i in test_list]
test_list.sort()

for i in train_list:
    template['training'].append({'image': './imagesTr/'+i, 'label': './labelsTr/'+i})


for i in test_list:
    template['test'].append({'image': './imagesTs/'+i, 'label': './labelsTs/'+i})

template['numTraining'] = len(train_list)
template['numTest'] = len(test_list)

with open('dataset.json', 'w') as f:
    json.dump(template, f)


#####

label_path = '/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/KidneyDiff_Dataset_Cleaned/0.5mm/label_6cls'
image_path = '/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/KidneyDiff_Dataset_Cleaned/0.5mm/img'

target_path = '/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task1008-YC_0.5mm_6cls_Remap_dataset'

import os, shutil
from tqdm import tqdm
os.makedirs(os.path.join(target_path, 'labelsTr'), exist_ok=True)
os.makedirs(os.path.join(target_path, 'imagesTr'), exist_ok=True)
os.makedirs(os.path.join(target_path, 'labelsTs'), exist_ok=True)
os.makedirs(os.path.join(target_path, 'imagesTs'), exist_ok=True)

# for train in tqdm(train_list):
#     label = os.path.join(label_path, train)
#     image = os.path.join(image_path, train)
#     shutil.copy(label, os.path.join(target_path, 'labelsTr'))
#     shutil.copy(image, os.path.join(target_path, 'imagesTr'))

for test in tqdm(test_list):
    label = os.path.join(label_path, test)
    image = os.path.join(image_path, test)
    shutil.copy(label, os.path.join(target_path, 'labelsTs'))
    shutil.copy(image, os.path.join(target_path, 'imagesTs'))