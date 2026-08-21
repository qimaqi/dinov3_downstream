#!/usr/bin/env bash
#SBATCH --job-name=run_dinov3_regr002_brainage_patch_reg_meanf_2
#SBATCH --output=sbatch_log/run_dinov3_regr002_brainage_patch_reg_meanf_2_%j.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=160GB

# source /usr/bmicnas03/data-biwi-01/qimaqi_data/data/miniconda3/etc/profile.d/conda.sh
# conda activate flexict
# set -euo pipefail

export JOB_SCRATCH=/scratch/qimaqi/${SLURM_JOB_ID}
export CONDA_PACK_IGNORE_EDITABLE=1
export CONDA_PACK_IGNORE_MISSING=1

source "$HOME/scripts/prepare_env_in_scratch.sh" flexict
source "$HOME/scripts/prepare_flexict_sources_in_scratch.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT_downstream/FlexiCT/
RAW_ROOT="/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/FOMO_Tasks/Task_3/Task_3"
TASK_NAME="REGR002_FOMO26_BrainAge"
SPLIT_ROOT="${SPLIT_ROOT:-${REPO_ROOT}/results/3d_regression/_splits}"
TRAIN_SPLIT="${TRAIN_SPLIT:-split_5fold_cv}"
TEST_SPLIT="${TEST_SPLIT:-test_5fold_cv}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/3d_regression/dinov3_slice_reg_5fold_brainage_patch_cls_meanf_2}"

export DINOV3_2D_CHECKPOINT=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT_downstream/FlexiCT/ckpts/pretrain_fomo_10k_pretrained_dinov3_dino_base_g8_e400_p16_mri_normalize/2D_final_model.pth
export FOMO_CACHE_ROOT="${FOMO_CACHE_ROOT:-${REPO_ROOT}/results/3d_regression/_cache/${TASK_NAME}/pretrain_fomo_10k_pretrained_dinov3_dino_base_g8_e400_p16_mri_normalize}"

cd "${REPO_ROOT}"
mkdir -p "${FOMO_CACHE_ROOT}"
mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${SPLIT_ROOT}"

EXTRA_ARGS=()
if [[ "${LORA_ENCODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--lora_encoder)
fi
if [[ "${UNFREEZE_ENCODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--unfreeze_encoder)
fi
EXTRA_ARGS+=(--mri_normalization robust_zscore --mri_low_percentile 0.5 --mri_high_percentile 99.5)

python downstream/3d_regression/dinov3_fintune_reg_from_slices.py \
  --task "${TASK_NAME}" \
  --raw_root "${RAW_ROOT}" \
  --split_root "${SPLIT_ROOT}" \
  --output_dir "${OUTPUT_ROOT}" \
  --dinov3_checkpoint "${DINOV3_2D_CHECKPOINT}" \
  --train_split "${TRAIN_SPLIT}" \
  --test_split "${TEST_SPLIT}" \
  --fold 2 \
  --slice_pool patch_cls \
  --modality_pool mean \
  --batch_size 1 \
  --slice_batch_size 32 \
  --epochs 120 \
  --cache \
  --cache_path "${FOMO_CACHE_ROOT}" \
  "${EXTRA_ARGS[@]}"
