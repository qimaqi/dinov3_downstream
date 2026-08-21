#!/usr/bin/env bash

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS-}"
source /usr/bmicnas03/data-biwi-01/qimaqi_data/data/miniconda3/etc/profile.d/conda.sh
conda activate flexict
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

RESULTS_ROOT="${REPO_ROOT}/results/3d_classify/dinov3_slice_cls_each_robust_zscore_split_v0_repeat10_val4/CLS002_FOMO26_Infarct"
WEIGHTS_ROOT="${SCRIPT_DIR}/weights_dinov3"
BACKBONE_SRC="${REPO_ROOT}/ckpts/pretrain_fomo_10k_pretrained_dinov3_dino_base_g8_e400_p16_mri_normalize/2D_final_model.pth"
BACKBONE_DST="${WEIGHTS_ROOT}/2D_final_model.pth"

SELECTED_FOLDS=(
    "fold_9"
    "fold_5"
    "fold_8"
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

export TASK1_BACKBONE_KIND="dinov3"
export DINOV3_2D_CHECKPOINT="${BACKBONE_DST}"
export TASK1_MODEL_ROOT="${WEIGHTS_ROOT}"
export TASK1_SELECTED_FOLDS="$(IFS=,; echo "${SELECTED_FOLDS[*]}")"
export TASK1_DINOV3_LAYER="${TASK1_DINOV3_LAYER:-12}"
export TASK1_DINOV3_FEATURE_DIM="${TASK1_DINOV3_FEATURE_DIM:-768}"
export TASK1_TARGET_SPACING_XY="${TASK1_TARGET_SPACING_XY:-1.0,1.0}"
export TASK1_PAD_HW="${TASK1_PAD_HW:-256,256}"
export TASK1_RESIZE_HW="${TASK1_RESIZE_HW:-none}"

echo "Prepared Task 1 DINOv3 ensemble weights:"
for fold in "${SELECTED_FOLDS[@]}"; do
    echo "  - ${WEIGHTS_ROOT}/${fold}/best.pt"
done
echo "Backbone kind: ${TASK1_BACKBONE_KIND}"
echo "Backbone: ${DINOV3_2D_CHECKPOINT}"
echo "TASK1_MODEL_ROOT=${TASK1_MODEL_ROOT}"
echo "TASK1_SELECTED_FOLDS=${TASK1_SELECTED_FOLDS}"
echo "TASK1_DINOV3_LAYER=${TASK1_DINOV3_LAYER}"
echo "TASK1_DINOV3_FEATURE_DIM=${TASK1_DINOV3_FEATURE_DIM}"
echo "TASK1_TARGET_SPACING_XY=${TASK1_TARGET_SPACING_XY}"
echo "TASK1_PAD_HW=${TASK1_PAD_HW}"
echo "TASK1_RESIZE_HW=${TASK1_RESIZE_HW}"

python "${SCRIPT_DIR}/predict.py" "$@"
