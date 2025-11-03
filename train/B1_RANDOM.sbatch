#!/bin/bash
#-------------------------------------------------------------------------------
#  SBATCH CONFIG
#-------------------------------------------------------------------------------
#SBATCH -N1
#SBATCH -n1
#SBATCH -c 32
#SBATCH --mem 300G
#SBATCH --time 2-00:00:00
#SBATCH --account=pal-lab
#SBATCH --gres gpu:1

## labels and outputs
#SBATCH --job-name=901_CICL_B1_RANDOM_fast
#SBATCH --output=901_CICL_B1_RANDOM_fast-%j.out

## notifications
#SBATCH --mail-user=esdft@missouri.edu
#SBATCH --mail-type=ALL
#-------------------------------------------------------------------------------

echo "### Starting at: $(date) ###"
echo "### Study ID: B1_RANDOM ###"

#-------------------------------------------------------------------------------
#  ENVIRONMENT SETUP
#-------------------------------------------------------------------------------

# --- Robust Conda Activation ---
# Initialize Conda environment for the shell (necessary in SLURM scripts)
CONDA_INIT_SCRIPT="/home/esdft/data/miniconda3/etc/profile.d/conda.sh"
if [ -f "$CONDA_INIT_SCRIPT" ]; then
    echo "Sourcing Conda initialization script..."
    source "$CONDA_INIT_SCRIPT"
else
    echo "ERROR: Conda initialization script not found at $CONDA_INIT_SCRIPT. Exiting."
    exit 1
fi

# Activate the specific environment
conda activate epai_final
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate conda environment 'epai_final'. Exiting."
    exit 1
fi

# --- Unique Output Directory ---
ABLATION_ROOT="/cluster/VAST/civalab/results/elham_results/ePAI_test/nnUNet_Ablation_Results_CICL"
STUDY_RESULTS_DIR="${ABLATION_ROOT}/B1_RANDOM"

# Export 'nnUNet_results' to override nnU-Net's default results path
export nnUNet_results="${STUDY_RESULTS_DIR}"
echo "### Results Directory (nnUNet_results): ${nnUNet_results} ###"

mkdir -p "${nnUNet_results}"

#-------------------------------------------------------------------------------
#  EXECUTION
#-------------------------------------------------------------------------------

## Navigate to training directory
cd /cluster/VAST/civalab/results/elham_results/ePAI_test/ePAI/train

## Set Curriculum Environment Variables (Configuration for the trainer)
export CURRICULUM_STRATEGY="RANDOM"

## Run nnUNet training
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train 901 3d_fullres all -tr nnUNetTrainer_Ablation_Curriculum

echo "### Finished at: $(date) ###"
