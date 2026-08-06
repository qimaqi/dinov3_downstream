# 3D Regression From FlexiCT 2D Slice Tokens

This directory contains a draft 3D regression interface for FOMO processed
regression tasks using the 2D `Flexi_CT_2D` encoder.

The model reads an Asparagus/FOMO processed 3D sample, converts the volume to a
sequence of 2D slices, extracts one FlexiCT CLS token per slice, then pools those
slice tokens into one volume token for regression.

Supported task:

- `REGR002_FOMO26_BrainAge`

Run:

```bash
bash downstream/3d_regression/run_fomo_regr002_brainage.sh
```

Enable LoRA adapters on the FlexiCT 2D backbone:

```bash
LORA_ENCODER=1 bash downstream/3d_regression/run_fomo_regr002_brainage.sh
```

Fully unfreeze the FlexiCT encoder:

```bash
UNFREEZE_ENCODER=1 bash downstream/3d_regression/run_fomo_regr002_brainage.sh
```

Default behavior freezes the FlexiCT encoder and trains only the cross-slice
pooler plus regression head. Metrics include MSE, RMSE, MAE, and R2.
