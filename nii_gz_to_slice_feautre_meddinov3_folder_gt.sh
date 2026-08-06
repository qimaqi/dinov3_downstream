#!/usr/bin/env bash
# set -euo pipefail
source /usr/bmicnas03/data-biwi-01/qimaqi_data/data/miniconda3/etc/profile.d/conda.sh
conda activate flexict

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/nii_gz_to_slice_feautre_meddinov3.py"

DEFAULT_INPUT_DIR="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset069_FOMO26_Meningioma_FLAIR/imagesTr"
DEFAULT_TARGET_ROOT="/usr/bmicnas01/data-biwi-01/ct_video_mae_bmicscratch/data/nnUNet_raw/Dataset069_FOMO26_Meningioma_FLAIR/imagesTr_feature_gt_meddinov3_10k"

if [[ $# -ge 1 ]]; then
  INPUT_DIR="$1"
else
  INPUT_DIR="${DEFAULT_INPUT_DIR}"
fi

if [[ $# -ge 2 ]]; then
  TARGET_ROOT="$2"
else
  TARGET_ROOT="${DEFAULT_TARGET_ROOT}"
fi

if [[ $# -ge 2 ]]; then
  shift 2
elif [[ $# -eq 1 ]]; then
  shift 1
fi

if [[ ! -d "${INPUT_DIR}" ]]; then
  echo "Input folder does not exist: ${INPUT_DIR}" >&2
  exit 1
fi

mkdir -p "${TARGET_ROOT}"
shopt -s nullglob

found=0
for nii_path in "${INPUT_DIR}"/*.nii.gz; do
  found=1
  case_name="$(basename "${nii_path}" .nii.gz)"
  case_output_dir="${TARGET_ROOT}/${case_name}"
  mkdir -p "${case_output_dir}"

  echo "Processing ${nii_path}"
  echo "  -> ${case_output_dir}"

  python "${PY_SCRIPT}" \
    --input_nii "${nii_path}" \
    --output_dir "${case_output_dir}" \
    --slice_method gt \
    "$@"
done

if [[ ${found} -eq 0 ]]; then
  echo "No .nii.gz files found in ${INPUT_DIR}" >&2
  exit 1
fi
