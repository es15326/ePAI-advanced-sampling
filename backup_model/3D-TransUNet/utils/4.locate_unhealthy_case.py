import nibabel as nib, numpy as np, glob, tqdm


path = '/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task1006-YC_3mm_6cls_Padded_dataset/labelsTs_3cls'

case_list = glob.glob(path + '/*.nii.gz')
case_list.sort()
print(len(case_list))
file = open('unhealthy_cases.txt', 'w')
for case in tqdm.tqdm(case_list):
    img = nib.load(case).get_fdata()
    if np.sum(img == 2) > 0:
        file.write(case.split('/')[-1] + '\n')
        print(case.split('/')[-1])

