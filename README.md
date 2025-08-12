# Action Recognition using Vision Transformer (ViT)

This project implements action recognition on the HMDB_simp dataset using Vision Transformer (ViT) and TimeSFormer models. All experiment outputs (logs, confusion matrices, model weights) are automatically saved in the Results folder.

## Structure
- configs/: Configuration files
- models/: Model implementations
- src/: Training, evaluation, dataset, transforms, and utilities
- Results/: Experiment outputs
- webapp.py: Streamlit web application
- generate_loss_curves.py: Loss visualization tool
- requirements.txt: Dependencies

## Output Saving
All logs, confusion matrices, and model weights (.pth) for each experiment are saved automatically in the corresponding Results subfolder.
