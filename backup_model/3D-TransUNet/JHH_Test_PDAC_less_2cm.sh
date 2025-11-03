

export nnUNet_codebase="/home/chenqi/bigpaper/FELIX/yucheng/3-Lustgarten/nnUNet"
export nnUNet_raw_data_base="/home/chenqi/bigpaper/data"
export nnUNet_preprocessed="/home/chenqi/bigpaper/code/nnUNet_preprocessed_jhh_train"
export RESULTS_FOLDER="./runs"

CONFIG='./configs/GeTU1009_3DTransUNet_epo500.yaml'
model_file='model_ep_500.model' #6

SAVE_FOLDER=./out/3DTransUnet_JHH_Test


DATA_PATH="/home/chenqi/bigpaper/data/Dataset1017_ePAI_3MM/imagesTr"
PATIENT_ID="JHH-Test-PDAC<2cm"
INPUT_CSV_PATH="/home/chenqi/bigpaper/code/ePAI-main-0416/train/input_csv/${PATIENT_ID}.csv"
OUTPUT_CSV_PATH="${SAVE_FOLDER}/out_csv/${PATIENT_ID}.csv"
OUTPUT_MASK_PATH="${SAVE_FOLDER}/predictions/"
MODEL_PATH="/home/chenqi/bigpaper/FELIX/yucheng/3-Lustgarten/3D-TransUNet/runs/UNet_IN_NANFang/Task1009_ePAI_3MM/nnUNetTrainerV2_DDP__nnUNetPlansv2.1/GeTU1009_3DTransUNet_epo500"

CUDA_VISIBLE_DEVICES=4 python3 inference.py \
--save_npz \
--config=$CONFIG \
--raw_data_dir=$DATA_PATH \
--disable_split \
--model_file=$model_file \
--model_latest \
--fold='all' \
--save_folder=$OUTPUT_MASK_PATH \
--input_csv $INPUT_CSV_PATH \
--output_csv $OUTPUT_CSV_PATH \
--output_folder_name $MODEL_PATH

