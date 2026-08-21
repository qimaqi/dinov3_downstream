#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="${1:-/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/FOMO_Tasks/Task_2/Task_2}"
PREPROCESSED_DIR="${DATASET_DIR}/preprocessed"
LABELS_DIR="${DATASET_DIR}/labels"
OUTPUT_DIR="${SCRIPT_DIR}/test_predictions"
CSV_PATH="${OUTPUT_DIR}/dice_scores.csv"
CONDA_SH="/usr/bmicnas03/data-biwi-01/qimaqi_data/data/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV_NAME="meddinov3"

export nnUNet_raw="./nnUNet_raw"
export nnUNet_preprocessed="./meddinov3_preprocessed"
export nnUNet_results="./dinov3_pretrained_results/"




if [[ ! -f "${CONDA_SH}" ]]; then
    echo "Missing conda setup script: ${CONDA_SH}" >&2
    exit 1
fi

source "${CONDA_SH}"
conda activate "${CONDA_ENV_NAME}"

if [[ ! -d "${PREPROCESSED_DIR}" ]]; then
    echo "Missing preprocessed directory: ${PREPROCESSED_DIR}" >&2
    exit 1
fi

if [[ ! -d "${LABELS_DIR}" ]]; then
    echo "Missing labels directory: ${LABELS_DIR}" >&2
    exit 1
fi

rm -rf "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

echo "dataset: ${DATASET_DIR}"
echo "predictions: ${OUTPUT_DIR}"

{
    echo "case_id,dice"
} > "${CSV_PATH}"

case_count=0
sum_dice=0

while IFS= read -r flair_path; do
    case_dir="$(dirname "${flair_path}")"
    rel_path="${case_dir#${PREPROCESSED_DIR}/}"
    case_id="${rel_path//\//_}"
    label_path="${LABELS_DIR}/${rel_path}/seg.nii.gz"
    pred_path="${OUTPUT_DIR}/${case_id}_pred.nii.gz"

    if [[ ! -f "${label_path}" ]]; then
        echo "Skipping ${case_id}: missing label ${label_path}" >&2
        continue
    fi

    echo "Running prediction for ${case_id}"
    python3 "${SCRIPT_DIR}/predict.py" \
        --flair "${flair_path}" \
        --output "${pred_path}"

    dice="$(
        python3 - "${pred_path}" "${label_path}" <<'PY'
import sys
import nibabel as nib
import numpy as np

pred = nib.load(sys.argv[1]).get_fdata()
gt = nib.load(sys.argv[2]).get_fdata()

pred_bin = pred > 0
gt_bin = gt > 0

pred_sum = int(pred_bin.sum())
gt_sum = int(gt_bin.sum())
intersection = int(np.logical_and(pred_bin, gt_bin).sum())

if pred_sum == 0 and gt_sum == 0:
    dice = 1.0
else:
    denom = pred_sum + gt_sum
    dice = 0.0 if denom == 0 else (2.0 * intersection) / denom

print(f"{dice:.6f}")
PY
    )"

    printf "%s,%s\n" "${case_id}" "${dice}" >> "${CSV_PATH}"
    printf "Dice %s: %s\n" "${case_id}" "${dice}"

    sum_dice="$(python3 - "${sum_dice}" "${dice}" <<'PY'
import sys
print(float(sys.argv[1]) + float(sys.argv[2]))
PY
    )"
    case_count=$((case_count + 1))
done < <(find "${PREPROCESSED_DIR}" -path '*/flair.nii.gz' | sort)

if [[ "${case_count}" -eq 0 ]]; then
    echo "No FLAIR cases found under ${PREPROCESSED_DIR}" >&2
    exit 1
fi

mean_dice="$(
    python3 - "${sum_dice}" "${case_count}" <<'PY'
import sys
print(f"{float(sys.argv[1]) / int(sys.argv[2]):.6f}")
PY
)"

echo "Processed ${case_count} cases"
echo "Mean Dice: ${mean_dice}"
echo "CSV saved to ${CSV_PATH}"
