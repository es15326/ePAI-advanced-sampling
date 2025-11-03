import os

root = '/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task1007-YC_3mm_6cls_Remap_dataset/labelsTs/'

case_list = os.listdir(root)

pdac_cases = [x for x in case_list if x.startswith('FELIX5') or x.startswith('FELIX-PDAC')]
cyst_cases = [x for x in case_list if x.startswith('FELIX-C')]
pnet_cases = [x for x in case_list if x.startswith('FELIX7')]
healthy_cases = [x for x in case_list if not (x.startswith('FELIX5') or x.startswith('FELIX-PDAC') or x.startswith('FELIX-C') or x.startswith('FELIX7'))]
print(len(pdac_cases)/len(case_list), len(cyst_cases)/len(case_list), len(pnet_cases)/len(case_list), len(healthy_cases)/len(case_list))
# 551 477 331 599
# 0.28 0.24 0.18 0.30