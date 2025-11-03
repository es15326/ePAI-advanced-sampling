import torch
fname = '/mnt/ccvl15/yucheng/KidneyDiff/3D-TransUNet/runs/UNet_IN_NANFang/Task901-Felix3mmSinglePhaseNoPNET/nnUNetTrainerV2_DDP__nnUNetPlansv2.1/GeTU901_max2former_ms-432_decl3_d192_mds_mw1.0_disds_bs2x8_adamw_warmup10_lr3e-4_masklossv1_masking_hungarian20_mhsa32_epo2500/all/model_final_checkpoint.model'
saved_model = torch.load(fname, map_location=torch.device('cpu'))
print(saved_model.keys())