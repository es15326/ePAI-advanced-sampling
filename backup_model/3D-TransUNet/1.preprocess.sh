# for name in `ls *`;do mv $name ${name%.nii.gz}_0000.nii.gz;done
export nnUNet_codebase="/home/chenqi/bigpaper/FELIX/yucheng/3-Lustgarten/nnUNet"
export nnUNet_raw_data_base="/home/chenqi/bigpaper/data"
export nnUNet_preprocessed="/home/chenqi/bigpaper/code/nnUNet_preprocessed_jhh_train"
export RESULTS_FOLDER="./runs"
nnUNet_plan_and_preprocess -t 1009

nnUNet_plan_and_preprocess -t 1009 -tl 32 -tf 32
