export nnUNet_codebase="../"
export nnUNet_raw_data_base="/data/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw"
export nnUNet_preprocessed="/data/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0002.datasets-nnunet-preprocessed"
export RESULTS_FOLDER="./runs"

CONFIG="configs/GeTU901_max2former_ms-432_decl3_d192_mds_mw1.0_disds_bs2x8_adamw_warmup10_lr3e-4_masklossv1_masking_hungarian20_mhsa32_epo250.yaml"
CUDA_VISIBLE_DEVICES=0 python3 inference.py --config=$CONFIG --raw_data_dir='/mnt/ccvl15/yucheng/KidneyDiff/Dataset/KidneyDiff_Dataset/nnUNet_Dataset/0001.datasets-nnunet-raw/nnUNet_raw_data/Task901-Felix3mmSinglePhaseNoPNET/imagesTs' --disable_split --fold='all' --model_latest