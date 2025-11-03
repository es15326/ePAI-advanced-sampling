import json



jieneng_path = '/data/jieneng/data/nnUNet_preprocessed/Task901-Felix3mmSinglePhaseNoPNET/dataset.json'

with open(jieneng_path, 'r') as f:
    data = json.load(f)

jntr_list = [d["image"].split('/')[-1] for d in data["training"]]
print('jieneng tr list', len(jntr_list))

yc_te_list = '/mnt/ccvl15/yucheng/KidneyDiff/DiffTumor/STEP3.SegmentationModel/utils_data/2.felix_test.list'

with open(yc_te_list, 'r') as f:
    ycte_list = [d.strip() for d in f.readlines()]

print('yc te list', len(ycte_list))


wx_te_list = '/mnt/ccvl15/yucheng/KidneyDiff/DiffTumor/STEP3.SegmentationModel/data_list/wx_test.list'

with open(wx_te_list, 'r') as f:
    wxte_list = [d.strip() for d in f.readlines()]

# print('yc tr list', len(yctr_list))



print('jieneng-tr in yc-te', len(set(jntr_list).intersection(set(ycte_list))), 'ratio:', len(set(jntr_list).intersection(set(ycte_list)))/len(ycte_list))
print('jieneng-tr in wx-te', len(set(jntr_list).intersection(set(wxte_list))), 'ratio:', len(set(jntr_list).intersection(set(ycte_list)))/len(wxte_list))


# path = '/data/jieneng/data/nnUNet_preprocessed/Task801_WORD/splits_final.pkl'

# import pickle
# with open(path, 'rb') as f:
#     splits = pickle.load(f)

# print(splits[0])