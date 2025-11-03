# Dynamically generated trainer alias for hyperparameter tuning
try:
    # Import the base trainer using relative import
    from .nnUNetTrainer_Targeted_FTL_SWA import nnUNetTrainer_Targeted_FTL_SWA
except ImportError:
    raise ImportError("Could not import base trainer nnUNetTrainer_Targeted_FTL_SWA. Ensure nnUNetTrainer_Targeted_FTL_SWA.py is in the same directory.")

# The class name must match the filename and the -tr argument
class nnUNetTrainer_Targeted_FTL_SWA__A0_7_B0_3_G1_333_SLR0_001(nnUNetTrainer_Targeted_FTL_SWA):
    # This alias inherits everything (including HParam reading from env vars) from the base trainer.
    pass
