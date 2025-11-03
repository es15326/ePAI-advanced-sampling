#!/bin/bash

# Define the base command and trainer class
DATASET_ID=901
CONFIG=3d_fullres
FOLDS=all
# The actual Python class containing the implementation
BASE_TRAINER=nnUNetTrainer_Targeted_FTL_SWA

# Define the absolute path to the nnU-Net trainer directory (ADJUST IF NECESSARY)
TRAINER_DIR="/cluster/VAST/civalab/results/elham_results/ePAI_test/ePAI/train/nnunetv2/training/nnUNetTrainer/"

# Verify the base trainer file exists
if [ ! -f "${TRAINER_DIR}/${BASE_TRAINER}.py" ]; then
    echo "ERROR: Base trainer file not found at ${TRAINER_DIR}/${BASE_TRAINER}.py"
    echo "Please verify the TRAINER_DIR variable in the script."
    exit 1
fi

# Function to launch a training job
launch_training() {
    GPU_ID=$1
    ALPHA=$2
    BETA=$3
    GAMMA=$4
    SWA_LR=$5

    # 1. Define the unique run name and file path
    # CRITICAL: Python class names cannot contain dots (.). We must replace them.
    # We use bash parameter expansion (//./_) to replace all dots with underscores (e.g., A0.7 becomes A0_7).
    SUFFIX="__A${ALPHA//./_}_B${BETA//./_}_G${GAMMA//./_}_SLR${SWA_LR//./_}"
    
    # The unique name used for the -tr argument AND the new class name
    RUN_NAME="${BASE_TRAINER}${SUFFIX}"
    TEMP_TRAINER_FILE="${TRAINER_DIR}/${RUN_NAME}.py"

    echo "------------------------------------------------------------------"
    echo "Launching Experiment: ${RUN_NAME} on GPU ${GPU_ID}"
    echo "Params: Alpha=${ALPHA}, Beta=${BETA}, Gamma=${GAMMA}, SWA_LR=${SWA_LR}"
    echo "------------------------------------------------------------------"

    # 2. Generate the temporary trainer file (Dynamic Alias)
    # This creates a new class named RUN_NAME that inherits from BASE_TRAINER.
    echo "Generating temporary trainer file: ${TEMP_TRAINER_FILE}"
    # Use Heredoc (cat <<EOF ...) to create the file content
    cat > ${TEMP_TRAINER_FILE} <<EOF
# Dynamically generated trainer alias for hyperparameter tuning
try:
    # Import the base trainer using relative import
    from .${BASE_TRAINER} import ${BASE_TRAINER}
except ImportError:
    raise ImportError("Could not import base trainer ${BASE_TRAINER}. Ensure ${BASE_TRAINER}.py is in the same directory.")

# The class name must match the filename and the -tr argument
class ${RUN_NAME}(${BASE_TRAINER}):
    # This alias inherits everything (including HParam reading from env vars) from the base trainer.
    pass
EOF

    # 3. Launch the training
    # We use the original numeric values (with dots) for the environment variables (read by the base class).
    # We use the unique RUN_NAME (with underscores) for the -tr argument (which now matches the generated class).
    # The '&' runs the command in the background (parallel execution).
    CUDA_VISIBLE_DEVICES=${GPU_ID} FTL_ALPHA=${ALPHA} FTL_BETA=${BETA} FTL_GAMMA=${GAMMA} SWA_LR=${SWA_LR} \
    nnUNetv2_train ${DATASET_ID} ${CONFIG} ${FOLDS} -tr ${RUN_NAME} &
    
    # Store the PID and the temp file path to clean up later
    PIDS+=($!)
    TEMP_FILES+=(${TEMP_TRAINER_FILE})
}

# --- Define Experiments ---
# Initialize arrays to track processes and files
PIDS=()
TEMP_FILES=()

# Exp 1: Baseline FTL (High Recall Focus, Alpha=0.7, Gamma=1.333)
launch_training 0 0.7 0.3 1.333 0.001

# Exp 2: Extreme Recall Focus (Alpha=0.85)
# Assuming you have a second GPU (ID 1)
launch_training 1 0.85 0.15 1.333 0.001

# Exp 3: High Gamma (Gamma=2.0)
# Assuming you have a third GPU (ID 2)
launch_training 2 0.7 0.3 2.0 0.001

# Example Exp 4: Balanced (Focal Dice equivalent, Alpha=0.5)
# Uncomment if you have a 4th GPU (ID 3)
# launch_training 3 0.5 0.5 1.333 0.001

# Wait for all background jobs to complete
echo "Waiting for experiments to complete..."
# Wait specifically for the PIDs launched by this script
wait "${PIDS[@]}"

# Clean up temporary files
echo "Cleaning up temporary trainer files..."
for file in "${TEMP_FILES[@]}"; do
    if [ -f "$file" ]; then
        # echo "Removing $file"
        rm "$file"
    fi
done

echo "All parallel experiments have finished."
