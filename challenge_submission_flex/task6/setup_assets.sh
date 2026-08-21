#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

mkdir -p "${SCRIPT_DIR}/dinov3_downstream/downstream" "${SCRIPT_DIR}/weights"
rsync -a --delete "${REPO_ROOT}/flexi_ct/" "${SCRIPT_DIR}/dinov3_downstream/flexi_ct/"
rsync -a --delete "${REPO_ROOT}/downstream/3d_classify/" "${SCRIPT_DIR}/dinov3_downstream/downstream/3d_classify/"
cp -f "${REPO_ROOT}/downstream/__init__.py" "${SCRIPT_DIR}/dinov3_downstream/downstream/__init__.py"
cp -f "${REPO_ROOT}/challenge_submission_flex/task1/weights_dinov3/2D_final_model.pth" "${SCRIPT_DIR}/weights/2D_final_model.pth"

echo "task6 assets staged under ${SCRIPT_DIR}"
