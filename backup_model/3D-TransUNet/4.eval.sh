# CONFIG='./configs/GeTU901_max2former_ms-432_decl3_d192_mds_mw1.0_disds_bs2x8_adamw_warmup10_lr3e-4_masklossv1_masking_hungarian20_mhsa32_epo250.yaml'
# RAW_DATA_DIR='/data/jieneng/data/nnUNet_raw_data/Task901-Felix3mmSinglePhaseNoPNET/imagesTs'

# CUDA_VISIBLE_DEVICES=0 python3 metric_measuring_dice_felix_2024lustgarten.py --config=$CONFIG --raw_data_dir=$RAW_DATA_DIR --disable_split --fold='all' --num_classes=5 --thres_cyst=30 --thres_pdac=30



# 測Sen.
# CONFIG='./configs/GeTU901_max2former_ms-432_decl3_d192_mds_mw1.0_disds_bs2x8_adamw_warmup10_lr3e-4_masklossv1_masking_hungarian20_mhsa32_epo250.yaml'
# RAW_DATA_DIR='/data/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task901-Felix3mmSinglePhaseNoPNET/imagesTs'

# CUDA_VISIBLE_DEVICES=0 nohup python3 metric_measuring_dice_felix_2024lustgarten.py --config=$CONFIG --raw_data_dir=$RAW_DATA_DIR --disable_split --fold='all' --num_classes=5 --thres_cyst=30 --thres_pdac=30 > Task901-GeTU901-Sen.log 2>&1 &

# 測Spe.
CONFIG='./configs/GeTU901_max2former_ms-432_decl3_d192_mds_mw1.0_disds_bs2x8_adamw_warmup10_lr3e-4_masklossv1_masking_hungarian20_mhsa32_epo250.yaml'
RAW_DATA_DIR='/data/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task901-Felix3mmSinglePhaseNoPNET/imagesTs'

CUDA_VISIBLE_DEVICES=0 nohup python3 metric_measuring_dice_felix_2024lustgarten.py --config=$CONFIG --raw_data_dir=$RAW_DATA_DIR --disable_split --fold='all' --num_classes=5 --thres_cyst=30 --thres_pdac=30 --normal > Task901-GeTU901-Spe.log 2>&1 &