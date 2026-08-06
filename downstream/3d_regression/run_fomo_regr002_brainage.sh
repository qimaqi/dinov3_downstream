#!/usr/bin/env bash
#SBATCH --job-name=run_fomo_regr002_brainage
#SBATCH --output=sbatch_log/run_fomo_regr002_brainage_%j.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160GB

source /usr/bmicnas03/data-biwi-01/qimaqi_data/data/miniconda3/etc/profile.d/conda.sh
conda activate flexict
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT_downstream/FlexiCT/
PROCESSED_ROOT="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FOMO_Challenge/processed_data"
TASK_NAME="REGR002_FOMO26_BrainAge"

# export FLEXICT_2D_CHECKPOINT="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT/ckpts/2D_final_model.pth"
export FLEXICT_2D_CHECKPOINT=/usr/bmicnas03/data-biwi-01/qimaqi_data/data/medical_journal/flexcit_outputs/leomed/pretrain_fomo_10k_pretrained_flexcit_base_g8_e200_p8_mri//2D_final_model.pth
export FLEXICT_3D_CHECKPOINT=/usr/bmicnas03/data-biwi-01/qimaqi_data/data/medical_journal/flexcit_outputs/leomed/pretrain_fomo_10k_pretrained_flexcit_base_g8_e200_p8_mri//3D_final_model.pth

export FOMO_CACHE_ROOT="${FOMO_CACHE_ROOT:-${PROCESSED_ROOT}/${TASK_NAME}/preprocessed_pretrain_fomo_10k_pretrained_flexcit_base_g8_e200_p8_mri}/"

cd "${REPO_ROOT}"
mkdir -p "${FOMO_CACHE_ROOT}"

RERUN=0
for arg in "$@"; do
  if [[ "${arg}" == "--rerun" ]]; then
    RERUN=1
  else
    echo "Unknown argument: ${arg}" >&2
    exit 1
  fi
done

RESULTS_ROOT="${REPO_ROOT}/results/3d_regression/fomo_slice_reg/${TASK_NAME}"
EXTRA_ARGS=()
if [[ "${LORA_ENCODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--lora_encoder)
fi
if [[ "${UNFREEZE_ENCODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--unfreeze_encoder)
fi

run_fold() {
  local fold="$1"
  local fold_dir="${RESULTS_ROOT}/fold_${fold}"
  local fold_args=()

  if [[ "${RERUN}" == "0" && -d "${fold_dir}" ]]; then
    fold_args+=(--resume)
    echo "[resume] fold ${fold}: detected existing results in ${fold_dir}"
  fi

  python downstream/3d_regression/fomo_finetune_reg_from_slices.py \
    --task "${TASK_NAME}" \
    --processed_root "${PROCESSED_ROOT}" \
    --checkpoint "${FLEXICT_2D_CHECKPOINT}" \
    --train_split split_80_10_10 \
    --test_split TEST_80_10_10 \
    --fold "${fold}" \
    --slice_pool cls \
    --modality_pool mean \
    --batch_size 1 \
    --slice_batch_size 32 \
    --epochs 50 \
    --cache \
    --cache_path "${FOMO_CACHE_ROOT}" \
    "${fold_args[@]}" \
    "${EXTRA_ARGS[@]}"
}

for fold in 0 1 2 3 4; do
  run_fold "${fold}"
done

# load from cached feature
# python downstream/3d_regression/fomo_finetune_reg_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split split_80_10_10   --test_split TEST_80_10_10   --fold 0   --slice_pool cls   --modality_pool mean   --batch_size 1   --slice_batch_size 32   --epochs 50   --from_cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"
# python downstream/3d_regression/fomo_finetune_reg_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split split_80_10_10   --test_split TEST_80_10_10   --fold 1   --slice_pool cls   --modality_pool mean   --batch_size 1   --slice_batch_size 32   --epochs 50   --from_cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"
# python downstream/3d_regression/fomo_finetune_reg_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split split_80_10_10   --test_split TEST_80_10_10   --fold 2   --slice_pool cls   --modality_pool mean   --batch_size 1   --slice_batch_size 32   --epochs 50   --from_cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"
# python downstream/3d_regression/fomo_finetune_reg_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split split_80_10_10   --test_split TEST_80_10_10   --fold 3   --slice_pool cls   --modality_pool mean   --batch_size 1   --slice_batch_size 32   --epochs 50   --from_cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"
# python downstream/3d_regression/fomo_finetune_reg_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split split_80_10_10   --test_split TEST_80_10_10   --fold 4   --slice_pool cls   --modality_pool mean   --batch_size 1   --slice_batch_size 32   --epochs 50   --from_cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"
