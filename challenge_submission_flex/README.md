# FOMO26 Challenge Submission

Containerized submission for the FOMO26 challenge using FlexiCT (DINOv3) backbone.

## Structure

```
submission/
├── setup_build.sh          # Run first: creates symlinks + weight dirs
├── README.md               # This file
├── task1/                  # Infarct Detection (Binary Classification)
│   ├── predict.py
│   ├── Apptainer.def
│   ├── requirements.txt
│   └── weights/            # Place: best.pt + 2D_final_model.pth
├── task2/                  # Meningioma Segmentation (nnU-Net)
│   ├── predict.py
│   ├── Apptainer.def
│   ├── requirements.txt
│   └── weights/            # Place: nnunet_model/ + 2D/3D_final_model.pth
├── task3/                  # Brain Age Estimation (Regression)
│   ├── predict.py
│   ├── Apptainer.def
│   ├── requirements.txt
│   └── weights/            # Place: best.pt + 2D_final_model.pth
├── task4/                  # Trigeminal Neuralgia Segmentation (nnU-Net)
│   ├── predict.py
│   ├── Apptainer.def
│   ├── requirements.txt
│   └── weights/            # Place: nnunet_model/ + 2D/3D_final_model.pth
├── task5/                  # Polymicrogyria Classification
│   ├── predict.py
│   ├── Apptainer.def
│   ├── requirements.txt
│   └── weights/            # Place: best.pt + 2D_final_model.pth
└── task6/                  # Linear Probing + Bias & Fairness (Embeddings)
    ├── predict.py          # Same container for Tasks 6 and 7
    ├── Apptainer.def
    ├── requirements.txt
    └── weights/            # Place: best.pt + 2D_final_model.pth
```

## Quick Start

```bash
# 1. Setup symlinks and directories
bash setup_build.sh

# 2. Copy your trained weights (example for task 1)
cp /path/to/infarct/best.pt task1/weights/best.pt
cp /path/to/2D_final_model.pth task1/weights/2D_final_model.pth

# 3. Build container
cd task1
apptainer build --fakeroot task1_infarct.sif Apptainer.def --arch amd64

# 4. Validate
# See: https://github.com/fomo26/container-validator
```

## Model Architecture

| Task | Architecture | Input | Output |
|------|-------------|-------|--------|
| 1. Infarct | FlexiCT 2D + SlicePool + Classifier | FLAIR, ADC, DWI, T2*, SWI | probability (.txt) |
| 2. Meningioma | nnU-Net + FlexiCT backbone (Primus) | FLAIR, DWI, T2*, SWI | binary mask (.nii.gz) |
| 3. Brain Age | FlexiCT 2D + SlicePool + Regressor | T1 | age value (.txt) |
| 4. Trigeminal | nnU-Net + FlexiCT backbone (Primus) | T2 | multiclass mask (.nii.gz) |
| 5. Polymicrogyria | FlexiCT 2D + SlicePool + Classifier | T1 | probability (.txt) |
| 6/7. Embedding | FlexiCT 2D + SlicePool (no head) | any MR | embedding (.npy) |

## Weight Files Needed

### Classification / Regression tasks (1, 3, 5, 6):
- `best.pt` — saved checkpoint with `{"model": state_dict, "args": training_args, "task": task_name}`
- `2D_final_model.pth` — pretrained FlexiCT 2D backbone

### Segmentation tasks (2, 4):
- `nnunet_model/` — full nnU-Net trained model folder containing:
  - `dataset.json`
  - `plans.json`
  - `fold_X/checkpoint_final.pth` (for each fold used)
- `2D_final_model.pth` — FlexiCT 2D backbone
- `3D_final_model.pth` — FlexiCT 3D backbone (if used)

## Notes

- Tasks 6 and 7 use the same container; submit the same .sif for both
- The `forward_features()` method extracts the volume token before the head (embedding)
- All containers use `pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime` as base
