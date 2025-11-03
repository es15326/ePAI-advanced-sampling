import pandas as pd

csv_file_path = '/mnt/ccvl15/yucheng/KidneyDiff/3D-TransUNet/error_analysis/PP/2024.0425.segresnet.tiny.epoch600.jhh.checkpoint.450/ID/FN.csv'
wx_patientIDs = pd.read_csv(csv_file_path)['patientID'].tolist()

csv_file_path = '/mnt/ccvl15/yucheng/KidneyDiff/3D-TransUNet/error_analysis/PP/Task1007-YC_3mm_6cls_Remap_dataset-GeTU1007_3DTransUNet_epo500/ID/FN.csv'
jn_patientIDs = pd.read_csv(csv_file_path)['patientID'].tolist()
print('FN', len(set(wx_patientIDs) & set(jn_patientIDs)) / len(jn_patientIDs))

csv_file_path = '/mnt/ccvl15/yucheng/KidneyDiff/3D-TransUNet/error_analysis/PP/2024.0425.segresnet.tiny.epoch600.jhh.checkpoint.450/ID/FP.csv'
wx_patientIDs = pd.read_csv(csv_file_path)['patientID'].tolist()

csv_file_path = '/mnt/ccvl15/yucheng/KidneyDiff/3D-TransUNet/error_analysis/PP/Task1007-YC_3mm_6cls_Remap_dataset-GeTU1007_3DTransUNet_epo500/ID/FP.csv'
jn_patientIDs = pd.read_csv(csv_file_path)['patientID'].tolist()
print('FP', len(set(wx_patientIDs) & set(jn_patientIDs)) / len(jn_patientIDs))
