
export nnUNet_N_proc_DA=10
export nnUNet_codebase="/home/chenqi/bigpaper/FELIX/yucheng/3-Lustgarten/nnUNet"
export nnUNet_raw_data_base="/home/chenqi/bigpaper/data"
export nnUNet_preprocessed="/home/chenqi/bigpaper/code/nnUNet_preprocessed_jhh_train"
export RESULTS_FOLDER="./runs"


CONFIG='./configs/GeTU1009_3DTransUNet_epo500.yaml'

echo $CONFIG

### unit test
fold='all'

# nnunet_use_progress_bar=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#         python3 -m torch.distributed.launch --master_port=4322 --nproc_per_node=8 \
#         ./train.py --fold=${fold} --config=$CONFIG --resume=''
# nnunet_use_progress_bar=1 CUDA_VISIBLE_DEVICES=0,1\
#         python3 -m torch.distributed.launch --master_port=4322 --nproc_per_node=6 \
#         ./train.py --fold=${fold} --continue_training --finetune --config=$CONFIG --resume='local_latest'

# nnunet_use_progress_bar=1 CUDA_VISIBLE_DEVICES=0,1,2,3\
#         python3 -m torch.distributed.launch --master_port=4322 --nproc_per_node=4 \
#         ./train.py --fold=${fold} --continue_training --config=$CONFIG --resume='local_latest'

nnunet_use_progress_bar=1 CUDA_VISIBLE_DEVICES=0,1,2,3\
        python3 -m torch.distributed.launch --master_port=4322 --nproc_per_node=4 \
        ./train.py --fold=${fold} --config=$CONFIG --continue_training --resume='local_latest'