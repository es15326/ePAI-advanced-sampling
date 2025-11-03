## Prerequisite

- 数据路径:

`/data/jieneng/data/nnUNet_raw_data/Task901-Felix3mmSinglePhaseNoPNET`

- Split: 

`/data/jieneng/data/nnUNet_preprocessed/Task901-Felix3mmSinglePhaseNoPNET/dataset.json`


## To Do
1. data preprocessing:
    1. nnUNet-v1 to preprocess FELIX big paper training set
2. inference:
    1. 3D-TransUNet inference on 901 PDAC-noPNet; Reproduce the result 
    2. 3D-TransUNet (trained on 901 PDAC-noPNet) inference on FELIX-BigPaper-TestSet
    3. Test-time augmentation + PostProcessing for the results of step 2.2
3. fine-tune in case step 2 results is not good enough
4. re-train in case step 3 results is not good enough

## Guide:

把github 3D-TransUNet的，配套这个config; 去做inference

1. 代码: https://github.com/Beckschen/3D-TransUNet/blob/main/inference.py
2. 需要的yaml文件: 刚发给你了
3. 环境: `/data/jieneng/software/anaconda/envs/nnunet` (或者按照github配置)
4. preprocessed data: `/data/jieneng/data/nnUNet_preprocessed/Task901-Felix3mmSinglePhaseNoPNET`
5. test-set: `/data/jieneng/data/nnUNet_raw_data/Task901-Felix3mmSinglePhaseNoPNET/imagesTs`
6. 模型: `/data/jieneng/results/felix_nnunet/UNet_IN_NANFang/Task901-Felix3mmSinglePhaseNoPNET/nnUNetTrainerV2_DDP__nnUNetPlansv2.1/GeTU901_max2former_ms-432_decl3_d192_mds_mw1.0_disds_bs2x8_adamw_warmup10_lr3e-4_masklossv1_masking_hungarian20_mhsa32_epo250`
7. inference指令

---

每次跑的时候都需要export一些环境变量:
```bash
export nnUNet_codebase="current_code_base" # 要改一下
export nnUNet_raw_data_base="/data2/jieneng/data"
export nnUNet_preprocessed="/data2/jieneng/data/nnUNet_preprocessed"
export RESULTS_FOLDER="/data/jieneng/results/felix_nnunet"

export nnUNet_raw_data_base="/data/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw"
export nnUNet_preprocessed="/data/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0002.datasets-nnunet-preprocessed"
export RESULTS_FOLDER="/data/yucheng/KidneyDiff/nnUnet_checkpoints/nnUnet_trained_models"

```

---
```bash
export nnUNet_codebase="../"
export nnUNet_raw_data_base="/data/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw"
export nnUNet_preprocessed="/data/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0002.datasets-nnunet-preprocessed"
export RESULTS_FOLDER="./output/"

CONFIG="configs/GeTU901_max2former_ms-432_decl3_d192_mds_mw1.0_disds_bs2x8_adamw_warmup10_lr3e-4_masklossv1_masking_hungarian20_mhsa32_epo250.yaml"
CUDA_VISIBLE_DEVICES=0 python3 Inference3D.py --config=$CONFIG --raw_data_dir='/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task901-Felix3mmSinglePhaseNoPNET/imagesTs' --disable_split --fold='all' --model_latest
```

---
我之前应该是拿这个文件处理过: 

`/data/jieneng/data/nnUNet_preprocessed/Task901-Felix3mmSinglePhaseNoPNET/get_test.py`