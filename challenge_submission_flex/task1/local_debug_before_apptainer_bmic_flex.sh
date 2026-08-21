#!/usr/bin/env bash

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS-}"
source /usr/bmicnas03/data-biwi-01/qimaqi_data/data/miniconda3/etc/profile.d/conda.sh
conda activate flexict
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RESULTS_ROOT="${REPO_ROOT}/results/3d_classify/fomo_slice_cls_each_robust_zscore_split50_50/CLS002_FOMO26_Infarct"
WEIGHTS_ROOT="${SCRIPT_DIR}/weights_flex"
BACKBONE_SRC="${REPO_ROOT}/ckpts/pretrain_fomo_100k_pretrained_flexcit_base_g8_e200_p8_mri_gram/2D_final_model_fomo100k_gram.pth"
BACKBONE_DST="${WEIGHTS_ROOT}/2D_final_model_fomo100k_gram.pth"

SELECTED_FOLDS=(
    "fold_0"
    "fold_1"
)

if [[ ! -f "${BACKBONE_SRC}" ]]; then
    echo "ERROR: Missing backbone checkpoint: ${BACKBONE_SRC}" >&2
    exit 1
fi

for fold in "${SELECTED_FOLDS[@]}"; do
    if [[ ! -f "${RESULTS_ROOT}/${fold}/best.pt" ]]; then
        echo "ERROR: Missing ensemble checkpoint: ${RESULTS_ROOT}/${fold}/best.pt" >&2
        exit 1
    fi
done

mkdir -p "${WEIGHTS_ROOT}"
cp -f "${BACKBONE_SRC}" "${BACKBONE_DST}"

for fold in "${SELECTED_FOLDS[@]}"; do
    mkdir -p "${WEIGHTS_ROOT}/${fold}"
    cp -f "${RESULTS_ROOT}/${fold}/best.pt" "${WEIGHTS_ROOT}/${fold}/best.pt"
    if [[ -f "${RESULTS_ROOT}/${fold}/metrics.json" ]]; then
        cp -f "${RESULTS_ROOT}/${fold}/metrics.json" "${WEIGHTS_ROOT}/${fold}/metrics.json"
    fi
done

export FLEXICT_2D_CHECKPOINT="${BACKBONE_DST}"
export TASK1_MODEL_ROOT="${WEIGHTS_ROOT}"
export TASK1_SELECTED_FOLDS="$(IFS=,; echo "${SELECTED_FOLDS[*]}")"

echo "Prepared Task 1 FlexiCT ensemble weights:"
for fold in "${SELECTED_FOLDS[@]}"; do
    echo "  - ${WEIGHTS_ROOT}/${fold}/best.pt"
done
echo "Backbone: ${FLEXICT_2D_CHECKPOINT}"
echo "TASK1_MODEL_ROOT=${TASK1_MODEL_ROOT}"
echo "TASK1_SELECTED_FOLDS=${TASK1_SELECTED_FOLDS}"

python "${SCRIPT_DIR}/predict.py" "$@"
