# FOMO CLS002 Infarct 3D Classification Pipeline

This document summarizes the code path launched by:

```bash
bash downstream/3d_classify/run_fomo_cls002_infarct.sh
```

## Entry Point

`downstream/3d_classify/run_fomo_cls002_infarct.sh` is an SLURM-ready launcher for the `CLS002_FOMO26_Infarct` classification task.

It requests:

- 1 node
- 0 GPUs in the SBATCH header, although the Python code will use CUDA if available
- 4 CPU cores
- 160 GB memory
- 24 hours runtime

The script activates the `flexict` conda environment, sets strict shell mode, resolves the repository root, exports the 2D FlexiCT checkpoint path, and then runs:

```bash
python downstream/3d_classify/fomo_finetune_cls_from_slices.py \
  --task CLS002_FOMO26_Infarct \
  --processed_root /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FOMO_Challenge/processed_data \
  --checkpoint /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT/ckpts/2D_final_model.pth \
  --train_split split_80_10_10 \
  --test_split TEST_80_10_10 \
  --fold 0 \
  --slice_pool attention \
  --modality_pool mean \
  --batch_size 1 \
  --slice_batch_size 32 \
  --epochs 50
```

Two optional environment variables change encoder tuning:

- `LORA_ENCODER=1`: adds `--lora_encoder`
- `UNFREEZE_ENCODER=1`: adds `--unfreeze_encoder`

If neither is set, the FlexiCT 2D encoder is frozen.

## Data Loading

The Python entry point is `downstream/3d_classify/fomo_finetune_cls_from_slices.py`.

For `CLS002_FOMO26_Infarct`, the task directory is:

```text
/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FOMO_Challenge/processed_data/CLS002_FOMO26_Infarct
```

The code reads:

- `dataset.json` for task metadata, especially `metadata.n_classes`
- `split_80_10_10.json` for train/validation folds
- `TEST_80_10_10.json` for test files

Fold `0` is selected from `split_80_10_10.json`.

Each sample file is loaded with `torch.load(path, map_location="cpu")` and is expected to contain at least:

```text
[image_tensor, label_tensor]
```

The image tensor must have shape:

```text
[C, H, W, D]
```

The label is converted to a scalar integer class id.

By default, each volume is resized to:

```text
[C, 128, 128, 128]
```

The resize path converts from `[C, H, W, D]` to PyTorch interpolation layout `[N, C, D, H, W]`, applies trilinear interpolation, then converts back to `[C, H, W, D]`.

## Model Architecture

The model class is `FlexiCTSliceVolumeClassifier`.

The architecture is:

```text
3D volume
  -> convert volume to 2D slices
  -> normalize each slice
  -> Flexi_CT_2D encoder
  -> one CLS token per slice
  -> cross-slice pooling
  -> linear classification head
  -> class logits
```

For this script, the main settings are:

- `slice_axis=-1`: slice along depth `D`
- `modality_pool=mean`: average input channels before slicing
- `slice_size=512`: resize each 2D slice to `512 x 512`
- `patch_size=8`: set FlexiCT backbone patch size
- `max_slices=128`: uniformly subsample to at most 128 slices
- `slice_pool=attention`: learn attention weights over slice CLS tokens

Volume-to-slice conversion starts from batch layout:

```text
[B, C, H, W, D]
```

With `modality_pool=mean`, channels are averaged:

```text
[B, C, H, W, D] -> [B, 1, H, W, D]
```

Then the volume is rearranged into 2D slice batches:

```text
[B, 1, H, W, D] -> [B, S, 1, H, W]
```

where `S` is the number of selected slices.

The slices are flattened to:

```text
[B * S, 1, H, W]
```

then resized to `512 x 512`, normalized per slice by subtracting the slice mean and dividing by the slice standard deviation, and passed through `Flexi_CT_2D`.

The 2D encoder returns a `cls_token` for every slice. These tokens are reshaped back to:

```text
[B, S, token_dim]
```

## Slice Pooling

The code supports four slice pooling modes:

- `mean`: average all slice tokens
- `max`: take the maximum over slices
- `attention`: score each slice token with a small MLP and compute a weighted sum
- `transformer`: prepend a learned CLS token and run a transformer encoder over slices

The CLS002 launcher uses `attention`.

The attention pooler is:

```text
LayerNorm(token_dim)
  -> Linear(token_dim, token_dim / 2)
  -> GELU
  -> Linear(token_dim / 2, 1)
  -> softmax over slices
  -> weighted sum of slice tokens
```

The pooled volume token is sent to:

```text
Linear(token_dim, n_classes)
```

## Encoder Tuning Modes

The script supports three encoder tuning modes:

- `frozen`: default; FlexiCT 2D encoder parameters have `requires_grad=False`
- `lora`: enabled by `LORA_ENCODER=1`; applies PEFT LoRA to the FlexiCT backbone
- `full`: enabled by `UNFREEZE_ENCODER=1`; trains all encoder parameters

Only one of LoRA and full unfreezing can be enabled at a time.

Default LoRA settings are:

- `lora_r=16`
- `lora_alpha=16`
- `lora_targets=qkv,proj`
- `lora_dropout=0.0`

## Training

The script constructs train, validation, and test `DataLoader`s with:

- `batch_size=1`
- `num_workers=4` by default
- shuffling enabled only for training

The optimizer is:

```text
AdamW(trainable_parameters, lr=1e-3, weight_decay=1e-4)
```

The loss is:

```text
CrossEntropyLoss
```

Training runs for 50 epochs. During each epoch:

1. Load a batch of 3D volumes and labels.
2. Convert each volume to 2D slices.
3. Encode slices with FlexiCT 2D.
4. Pool slice CLS tokens into one volume token.
5. Predict class logits.
6. Compute cross-entropy loss.
7. Backpropagate through trainable parameters.
8. Evaluate on the validation set.

When the encoder is frozen, the encoder is kept in eval mode during training and slice encoding runs under `torch.no_grad()`.

The best checkpoint is selected by lowest validation loss.

## Evaluation

Validation and test evaluation compute:

- mean cross-entropy loss
- AUC
- balanced accuracy

For binary classification, AUC is computed from the probability of class `1`.

For multiclass classification, AUC uses one-vs-rest macro averaging.

If AUC cannot be computed, for example because only one class is present in a split, the code returns `NaN`.

## Outputs

Outputs are saved under:

```text
results/3d_classify/fomo_slice_cls/CLS002_FOMO26_Infarct
```

The files are:

- `best.pt`: model state dict, parsed arguments, task name, best validation loss, and test metrics
- `metrics.json`: best validation loss and final test metrics

## Important Notes

- The shell script SBATCH header requests `--gres=gpu:0`, but the Python code will use CUDA whenever `torch.cuda.is_available()` and `--cpu` is not set.
- The default run freezes the 2D FlexiCT encoder, so only the attention slice pooler and classification head are trained.
- `slice_batch_size=32` controls how many 2D slices are encoded at once inside a 3D volume batch. This is separate from the volume-level `batch_size=1`.
- The pipeline treats the 3D task as slice-token aggregation: there is no native 3D convolutional encoder in this classification code path.
