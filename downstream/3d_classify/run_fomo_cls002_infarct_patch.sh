#!/usr/bin/env bash
#SBATCH --job-name=run_fomo_cls002_infarct_patch
#SBATCH --output=sbatch_log/run_fomo_cls002_infarct_patch_%j.out
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
TASK_NAME="CLS002_FOMO26_Infarct"

# export FLEXICT_2D_CHECKPOINT="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT/ckpts/2D_final_model.pth"
export FLEXICT_2D_CHECKPOINT=/usr/bmicnas03/data-biwi-01/qimaqi_data/data/medical_journal/flexcit_outputs/leomed/pretrain_fomo_10k_pretrained_flexcit_base_g8_e200_p8_mri//2D_final_model.pth
export FLEXICT_3D_CHECKPOINT=/usr/bmicnas03/data-biwi-01/qimaqi_data/data/medical_journal/flexcit_outputs/leomed/pretrain_fomo_10k_pretrained_flexcit_base_g8_e200_p8_mri//3D_final_model.pth


export FOMO_CACHE_ROOT="${FOMO_CACHE_ROOT:-${PROCESSED_ROOT}/${TASK_NAME}/preprocessed_pretrain_fomo_10k_pretrained_flexcit_base_g8_e200_p8_mri}/"

cd "${REPO_ROOT}"
mkdir -p "${FOMO_CACHE_ROOT}"

EXTRA_ARGS=()
if [[ "${LORA_ENCODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--lora_encoder)
fi
if [[ "${UNFREEZE_ENCODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--unfreeze_encoder)
fi


# start cache feature
python downstream/3d_classify/fomo_finetune_cls_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split split_80_10_10   --test_split TEST_80_10_10   --fold 0   --slice_pool patch   --modality_pool mean   --batch_size 1   --slice_batch_size 32   --epochs 50   --cache   --cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"

python downstream/3d_classify/fomo_finetune_cls_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split split_80_10_10   --test_split TEST_80_10_10   --fold 1   --slice_pool patch   --modality_pool mean   --batch_size 1   --slice_batch_size 32   --epochs 50   --cache   --cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"

python downstream/3d_classify/fomo_finetune_cls_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split split_80_10_10   --test_split TEST_80_10_10   --fold 2   --slice_pool patch   --modality_pool mean   --batch_size 1   --slice_batch_size 32   --epochs 50   --cache   --cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"

python downstream/3d_classify/fomo_finetune_cls_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split split_80_10_10   --test_split TEST_80_10_10   --fold 3   --slice_pool patch   --modality_pool mean   --batch_size 1   --slice_batch_size 32   --epochs 50   --cache   --cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"


python downstream/3d_classify/fomo_finetune_cls_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split split_80_10_10   --test_split TEST_80_10_10   --fold 4   --slice_pool patch   --modality_pool mean   --batch_size 1   --slice_batch_size 32   --epochs 50   --cache   --cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"




# load from cached feature
# python downstream/3d_classify/fomo_finetune_cls_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split split_80_10_10   --test_split TEST_80_10_10   --fold 0   --slice_pool patch   --modality_pool mean   --batch_size 1   --slice_batch_size 32   --epochs 50   --from_cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"


# python downstream/3d_classify/fomo_finetune_cls_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split split_80_10_10   --test_split TEST_80_10_10   --fold 1   --slice_pool patch   --modality_pool mean   --batch_size 1   --slice_batch_size 32   --epochs 50   --from_cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"
