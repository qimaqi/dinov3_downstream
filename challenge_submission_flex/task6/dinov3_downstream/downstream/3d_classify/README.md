# 3D Classification From FlexiCT 2D Slice Tokens

This directory contains a draft 3D classification interface for FOMO processed
classification tasks using the 2D `Flexi_CT_2D` encoder.

The model reads an Asparagus/FOMO processed 3D sample, converts the volume to a
sequence of 2D slices, extracts one FlexiCT CLS token per slice, then pools those
slice tokens into one volume token for classification.

Supported tasks:

- `CLS002_FOMO26_Infarct`
- `CLS003_FOMO26_Polymicrogyria`

Run examples:

```bash
bash downstream/3d_classify/run_fomo_cls002_infarct.sh
bash downstream/3d_classify/run_fomo_cls003_polymicrogyria.sh
```

Enable LoRA adapters on the FlexiCT 2D backbone:

```bash
LORA_ENCODER=1 bash downstream/3d_classify/run_fomo_cls002_infarct.sh
```

Fully unfreeze the FlexiCT encoder:

```bash
UNFREEZE_ENCODER=1 bash downstream/3d_classify/run_fomo_cls002_infarct.sh
```

Default behavior freezes the FlexiCT encoder and trains only the cross-slice
pooler plus classifier head.
