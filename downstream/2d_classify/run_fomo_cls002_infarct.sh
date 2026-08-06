#!/usr/bin/env bash
#SBATCH --job-name=process_fomo_all
#SBATCH --output=sbatch_log/process_fomo_all_%j.out
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --mem=160GB


source /usr/bmicnas03/data-biwi-01/qimaqi_data/data/miniconda3/etc/profile.d/conda.sh
conda activate flexict
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export FLEXICT_2D_CHECKPOINT="/usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FlexiCT/ckpts/2D_final_model.pth"

cd "${REPO_ROOT}"

python downstream/2d_classify/fomo_finetune_cls_from_slices.py \
  --task CLS002_FOMO26_Infarct \
  --processed_root /usr/bmicnas02/data-biwi-01/qimaqi_data/workspace/medical_journal/FOMO_Challenge/processed_data \
  --checkpoint "${FLEXICT_2D_CHECKPOINT}" \
  --train_split split_80_10_10 \
  --test_split TEST_80_10_10 \
  --fold 0 \
  --slice_pool attention \
  --modality_pool mean \
  --batch_size 1 \
  --slice_batch_size 32 \
  --epochs 50 \
  --unfreeze_encoder