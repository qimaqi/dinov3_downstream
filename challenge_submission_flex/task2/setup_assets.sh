#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDDINOV3_SRC="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/MedDINOv3/nnUNet"
MODEL_SRC="/usr/bmicnas03/data-biwi-01/qimaqi_data/data/medical_journal/meddinov3_results/Dataset069_FOMO26_Meningioma_FLAIR/dinov3_base_primus_Trainer_freeze_multi__nnUNetPlans__2d"
PRETRAIN_SRC="/usr/bmicnas02/data-biwi-01/fm_originalzoo/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

mkdir -p "${SCRIPT_DIR}/weights/nnunet_model/fold_0"
rm -rf "${SCRIPT_DIR}/nnUNet"
rsync -a --delete "${MEDDINOV3_SRC}/" "${SCRIPT_DIR}/nnUNet/"
cp -f "${PRETRAIN_SRC}" "${SCRIPT_DIR}/weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
cp -f "${MODEL_SRC}/dataset.json" "${SCRIPT_DIR}/weights/nnunet_model/dataset.json"
cp -f "${MODEL_SRC}/dataset_fingerprint.json" "${SCRIPT_DIR}/weights/nnunet_model/dataset_fingerprint.json"
cp -f "${MODEL_SRC}/plans.json" "${SCRIPT_DIR}/weights/nnunet_model/plans.json"
cp -f "${MODEL_SRC}/fold_0/checkpoint_best.pth" "${SCRIPT_DIR}/weights/nnunet_model/fold_0/checkpoint_best.pth"

echo "task2 assets staged under ${SCRIPT_DIR}"
