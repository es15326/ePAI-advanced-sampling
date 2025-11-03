#export NNUNET_NEG_SAMPLES=200
#export NNUNET_FG_OVERSAMPLE=0.5

#nnUNetv2_train 901 3d_fullres all -tr nnUNetTrainer_FlexibleLesionSampler



export CURRICULUM_STRATEGY="STEPPED_E2H"
export CURRICULUM_METRIC="S10_Composite"
export CURRICULUM_STEPS="5"
nnUNetv2_train 901 3d_fullres all -tr nnUNetTrainer_Ablation_Curriculum
