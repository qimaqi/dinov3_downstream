# 2D Classification Pipeline

This directory implements a frozen-feature classification workflow for
`Flexi_CT_2D`. The pipeline has two executable stages:

1. extract and cache image-level features from CuriaBench datasets;
2. train linear heads on fixed fractions of those cached features and report
   data-efficiency metrics.

The scripts are intended to be run from the FlexiCT repository root:

```bash
cd /path/to/FlexiCT
```

## Files

- `extract_features_for_sweep.py`: loads `Flexi_CT_2D`, extracts fixed features
  for supported datasets, and saves NumPy feature/label arrays.
- `data_efficiency_sweep.py`: loads cached arrays, trains linear classifiers at
  several training-set fractions and learning rates, and writes metrics to CSV.
- `README.md`: quick reproduction commands.
- `config/*.yaml`: reference experiment configurations copied from the original
  classification runs. The active reproduction scripts do not parse these YAML
  files except for checking that the config directory exists during feature
  extraction.

## Supported Datasets

The executable scripts currently support these CuriaBench datasets:

| CLI name | Hugging Face config | Classes | Case ID column | Feature pooling |
| --- | --- | ---: | --- | --- |
| `kits` | `kits` | 2 | `series_id` | masked patch mean if a mask exists, otherwise CLS token |
| `deep-lesion` | `deep-lesion-site` | 8 | `series_id` | masked patch mean if a mask exists, otherwise CLS token |
| `covidx` | `covidx-ct` | 3 | none | concatenate CLS token and mean patch token |

The `config/` folder also contains reference settings for LUNA/LUNA16-3D and
baseline or alternative-model runs (`other-*.yaml`), but those variants are not
wired into the current `DATASETS` list used by the two pipeline scripts.

## Checkpoint Resolution

Feature extraction constructs the model with:

```python
Flexi_CT_2D(checkpoint_path=resolve_flexict_checkpoint("2d", checkpoint))
```

Checkpoint lookup order is:

1. `--checkpoint /path/to/checkpoint.pth`
2. `FLEXICT_CHECKPOINT`
3. `FLEXICT_2D_CHECKPOINT`

No private checkpoint path is bundled, so one of these must be set before
feature extraction.

Example:

```bash
export FLEXICT_2D_CHECKPOINT=/path/to/ct_2d_teacher.pth
```

The extractor also downloads CuriaBench datasets and the
`AutoImageProcessor.from_pretrained("raidium/curia", trust_remote_code=True)`
processor from Hugging Face on first use. On shared or offline systems, set a
Hugging Face cache location first:

```bash
export HF_HOME=/path/to/huggingface_cache
```

## Stage 1: Feature Extraction

Run all supported datasets:

```bash
python downstream/2d_classify/extract_features_for_sweep.py \
  --dataset all \
  --output_dir features/2d_classify \
  --batch_size 64 \
  --patch_size 8
```

Run one dataset:

```bash
python downstream/2d_classify/extract_features_for_sweep.py \
  --dataset kits \
  --output_dir features/2d_classify
```

### Extraction Flow

For each requested dataset and each available split among `train`, `val`, and
`test`, the extractor:

1. loads the CuriaBench split with `datasets.load_dataset`;
2. clips each image array to `[-1000, 1000]`;
3. preprocesses images with the `raidium/curia` image processor;
4. converts 3-channel processor output to 1 channel by channel averaging;
5. forwards the batch through `Flexi_CT_2D` in evaluation mode;
6. builds the cached feature vector according to the dataset feature mode;
7. writes features and labels to disk, plus case IDs when available.

The script sets the backbone patch size by calling `set_patch_size()` on any
module that exposes it and by updating `model.patch_size` when present.

### Feature Modes

`mask_or_cls` is used by `kits` and `deep-lesion`.

- If a non-empty mask is present, the mask is resized to the square patch-token
  grid with nearest-neighbor interpolation.
- Patch tokens inside the mask are averaged.
- If no usable mask exists, the CLS token is used.

`cls_patch_mean` is used by `covidx`.

- The CLS token is concatenated with the mean of all patch tokens.

`cls` exists as a helper mode in the extractor, but no currently registered
dataset uses it.

### Cached Outputs

By default, features are saved under:

```text
features/2d_classify/flexi/{dataset}/
```

Expected files are:

```text
{split}_features.npy
{split}_labels.npy
{split}_case_ids.npy   # only when the dataset has series_id
```

Existing `{split}_features.npy` and `{split}_labels.npy` files cause that split
to be skipped.

## Stage 2: Data-Efficiency Sweep

Run the full sweep:

```bash
python downstream/2d_classify/data_efficiency_sweep.py \
  --features_dir features/2d_classify \
  --output_csv results/2d_classify/data_efficiency_results.csv
```

Debug run:

```bash
python downstream/2d_classify/data_efficiency_sweep.py \
  --dataset kits \
  --fractions 1.0 \
  --lrs 0.01 \
  --epochs 1 \
  --n_bootstrap 10
```

### Sweep Defaults

- datasets: `kits`, `deep-lesion`, `covidx`
- training fractions: `0.01`, `0.05`, `0.10`, `0.25`, `0.50`, `1.00`
- learning rates: `0.001`, `0.005`, `0.01`, `0.05`, `0.1`, `0.5`, `1.0`, `5.0`
- epochs: `200` for all three supported datasets
- batch size: `64`
- optimizer: SGD with momentum `0.9`
- learning-rate schedule: cosine annealing over all training steps
- bootstrap samples for confidence intervals: `10000`
- random seed: `42`

Learning rates are scaled internally as:

```python
effective_lr = lr * batch_size / 256.0
```

### Training and Selection

For each dataset and fraction:

1. load `train_features.npy`, `train_labels.npy`, `test_features.npy`, and
   `test_labels.npy`;
2. skip the dataset if any required cache file is missing;
3. stratify-subsample the training set when `fraction < 1.0`;
4. train one `torch.nn.Linear` classifier per learning rate;
5. evaluate after every epoch on the test set;
6. keep the epoch and learning rate with the best test AUC;
7. compute final AUC, balanced accuracy, and bootstrap confidence intervals.

For datasets with `test_case_ids.npy`, test probabilities are averaged by case
ID before metrics are computed. This affects `kits` and `deep-lesion`.

## Outputs

The sweep writes:

```text
results/2d_classify/data_efficiency_results.csv
```

Columns:

```text
model,dataset,fraction,auc,auc_lo,auc_hi,bal_acc,bal_acc_lo,bal_acc_hi
```

Per-fraction probabilities are saved under:

```text
results/2d_classify/sweep_probabilities/flexi/{dataset}/frac_{fraction}.npz
```

Each `.npz` contains:

- `probs`: best predicted probabilities after optional case aggregation;
- `labels`: labels aligned with `probs`;
- `best_lr`: selected learning rate for that dataset/fraction.

## Practical Notes

- The sweep reports test-set model selection because it chooses the best epoch
  and learning rate by test AUC. Treat the output as reproduction/benchmark
  code, not as a validation-holdout protocol.
- The extractor maps all available `train`, `val`, and `test` splits, but the
  sweep uses only `train` and `test`.
- `val` caches may still be useful for extending the pipeline to validation-set
  learning-rate selection.
- Very small fractions are skipped when they would contain fewer than
  `max(2, n_classes)` samples.
