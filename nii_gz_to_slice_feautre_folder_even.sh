#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/nii_gz_to_slice_feautre.py"

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <nii_gz_folder> <target_root> [extra args passed to nii_gz_to_slice_feautre.py]" >&2
  exit 1
fi

INPUT_DIR="$1"
TARGET_ROOT="$2"
shift 2

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

  python "${PY_SCRIPT}"     --input_nii "${nii_path}"     --output_dir "${case_output_dir}"     --slice_method even     "$@"
done

if [[ ${found} -eq 0 ]]; then
  echo "No .nii.gz files found in ${INPUT_DIR}" >&2
  exit 1
fi
