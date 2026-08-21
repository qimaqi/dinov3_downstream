#!/usr/bin/env bash
#SBATCH --job-name=run_fomo_cls002_infarct_patch_cls
#SBATCH --output=sbatch_log/run_fomo_cls002_infarct_patch_cls_%j.out
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
TRAIN_SPLIT="${TRAIN_SPLIT:-split_v0_repeat10_val4_balanced}"
TEST_SPLIT="${TEST_SPLIT:-TEST_split_v0_4cases_balanced}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/3d_classify/fomo_slice_cls_each_robust_zscore_split_v0_repeat10_val4}"

# export FLEXICT_2D_CHECKPOINT="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT/ckpts/2D_final_model.pth"
export FLEXICT_2D_CHECKPOINT=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT_downstream/FlexiCT/ckpts/pretrain_fomo_100k_pretrained_flexcit_base_g8_e200_p8_mri_gram/2D_final_model_fomo100k_gram.pth

# /usr/bmicnas03/data-biwi-01/qimaqi_data/data/medical_journal/flexcit_outputs/leomed/pretrain_fomo_10k_pretrained_flexcit_base_g8_e200_p8_mri//2D_final_model.pth
export FLEXICT_3D_CHECKPOINT=/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT_downstream/FlexiCT/ckpts/pretrain_fomo_100k_pretrained_flexcit_base_g8_e200_p8_mri_gram/2D_final_model_fomo100k_gram.pth

# /usr/bmicnas03/data-biwi-01/qimaqi_data/data/medical_journal/flexcit_outputs/leomed/pretrain_fomo_10k_pretrained_flexcit_base_g8_e200_p8_mri//3D_final_model.pth

export FOMO_CACHE_ROOT="${FOMO_CACHE_ROOT:-${PROCESSED_ROOT}/${TASK_NAME}/pretrain_fomo_100k_pretrained_flexcit_base_g8_e200_p8_mri_gram}/"

cd "${REPO_ROOT}"
mkdir -p "${FOMO_CACHE_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

EXTRA_ARGS=()
if [[ "${LORA_ENCODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--lora_encoder)
fi
if [[ "${UNFREEZE_ENCODER:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--unfreeze_encoder)
fi
EXTRA_ARGS+=(--mri_normalization robust_zscore --mri_low_percentile 0.5 --mri_high_percentile 99.5)

for fold in {0..9}; do
  python downstream/3d_classify/fomo_finetune_cls_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --output_dir "${OUTPUT_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split "${TRAIN_SPLIT}"   --test_split "${TEST_SPLIT}"   --fold "${fold}"   --slice_pool patch_cls   --modality_pool each   --batch_size 1   --slice_batch_size 32   --epochs 200   --cache   --cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"
done

python downstream/3d_classify/rank_cls_results.py \
  --results_root "${OUTPUT_ROOT}" \
  --task "${TASK_NAME}" \
  --top_k 5




# load from cached feature
# python downstream/3d_classify/fomo_finetune_cls_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --output_dir "${OUTPUT_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split split_80_10_10   --test_split TEST_80_10_10   --fold 0   --slice_pool patch_cls   --modality_pool each   --batch_size 1   --slice_batch_size 32   --epochs 50   --from_cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"


# python downstream/3d_classify/fomo_finetune_cls_from_slices.py   --task "${TASK_NAME}"   --processed_root "${PROCESSED_ROOT}"   --output_dir "${OUTPUT_ROOT}"   --checkpoint "${FLEXICT_2D_CHECKPOINT}"   --train_split split_80_10_10   --test_split TEST_80_10_10   --fold 1   --slice_pool patch_cls   --modality_pool each   --batch_size 1   --slice_batch_size 32   --epochs 50   --from_cache_path "${FOMO_CACHE_ROOT}"   "${EXTRA_ARGS[@]}"
