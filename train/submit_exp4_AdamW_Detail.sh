#!/bin/bash
#-------------------------------------------------------------------------------
# SBATCH CONFIG
#-------------------------------------------------------------------------------
#SBATCH -N1
#SBATCH -n1
#SBATCH -c 32
#SBATCH --mem 300G
#SBATCH --time 2-00:00:00
#SBATCH --account=pal-lab
#SBATCH --gres gpu:1

## labels and outputs
#SBATCH --job-name=901_AdamW_Detail
#SBATCH --output=901_AdamW_Detail-%j.out

## notifications
#SBATCH --mail-user=esdft@missouri.edu
#SBATCH --mail-type=ALL
#-------------------------------------------------------------------------------

echo "### Starting at: $(date) ###"

# ------------------------------------------------------------------------------
# EXPERIMENT CONFIGURATION (Comprehensive Tuning)
# ------------------------------------------------------------------------------

# FTL Parameters (Baseline)
export FTL_ALPHA=0.7
export FTL_BETA=0.3
export FTL_GAMMA=1.333
export SWA_LR=0.001

# Optimizer Parameters (Switch to AdamW, standard LR)
export OPTIMIZER_TYPE="AdamW"
export INIT_LR=0.01
export WEIGHT_DECAY=3e-5

# Augmentation/Sampling (Tighter Scaling)
export OVERSAMPLE_FG=0.50
export TARGET_RATIO=0.75
export AUG_SCALE_MIN=0.80 # Default 0.70 (Less distortion)
export AUG_SPATIAL_P=0.25

# Define Constants
DATASET_ID=901
CONFIG=3d_fullres
FOLDS=all
BASE_TRAINER="nnUNetTrainer_Targeted_FTL_SWA"
TRAINER_DIR="/cluster/VAST/civalab/results/elham_results/ePAI_test/ePAI/train/nnunetv2/training/nnUNetTrainer/"
WORK_DIR="/cluster/VAST/civalab/results/elham_results/ePAI_test/ePAI/train"

# Generate the unique Run Name (replacing dots with underscores)
# Create a unique suffix based on key parameters for clarity.
SUFFIX="__${OPTIMIZER_TYPE}_LR${INIT_LR//./_}_S${AUG_SCALE_MIN//./_}"
RUN_NAME="${BASE_TRAINER}${SUFFIX}"
TEMP_TRAINER_FILE="${TRAINER_DIR}/${RUN_NAME}.py"

echo "Run Name: ${RUN_NAME}"

# ------------------------------------------------------------------------------
# ENVIRONMENT SETUP & CLEANUP TRAP
# ------------------------------------------------------------------------------

## Activate environment
echo "Activating environment..."
source activate epai_final

## Navigate to training directory
cd ${WORK_DIR}

# Function to clean up the temporary file
cleanup() {
    echo "Cleaning up temporary trainer file: ${TEMP_TRAINER_FILE}"
    if [ -f "$TEMP_TRAINER_FILE" ]; then
        rm "$TEMP_TRAINER_FILE"
    fi
    echo "### Finished at: $(date) ###"
}
trap cleanup EXIT

# ------------------------------------------------------------------------------
# DYNAMIC ALIAS GENERATION
# ------------------------------------------------------------------------------

echo "Generating temporary trainer file: ${TEMP_TRAINER_FILE}"
# Create the dynamic alias file
cat > ${TEMP_TRAINER_FILE} <<EOF
# Dynamically generated trainer alias
try:
    from .${BASE_TRAINER} import ${BASE_TRAINER}
except ImportError:
    raise ImportError("Could not import base trainer ${BASE_TRAINER}.")

class ${RUN_NAME}(${BASE_TRAINER}):
    pass
EOF

# ------------------------------------------------------------------------------
# EXECUTE TRAINING
# ------------------------------------------------------------------------------

echo "Starting nnUNet training..."
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLDS} -tr ${RUN_NAME}
