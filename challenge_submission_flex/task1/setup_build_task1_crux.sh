#!/usr/bin/env bash
# Build the Apptainer container for Task 1 (Infarct Detection)
# Requires: conda env "apptainer-test" with apptainer and fakeroot installed
# Usage: bash setup_build_task1_crux.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SIF_NAME="task1_infarct.sif"

# Activate conda environment with apptainer + fakeroot
eval "$(conda shell.bash hook)"
conda activate apptainer-test

echo "=== Building Task 1 container ==="
echo "  Working dir: ${SCRIPT_DIR}"
echo "  Definition:  Apptainer.def"
echo "  Output:      ${SIF_NAME}"
echo "  Conda env:   apptainer-test"
echo "  Apptainer:   $(which apptainer)"
echo ""

# Verify repo structure is accessible
if [ ! -d "../../flexi_ct" ]; then
    echo "ERROR: Cannot find ../../flexi_ct — run this script from within the repo."
    exit 1
fi

REQUIRED_FILES=(
    "./weights/2D_final_model_fomo100k_gram.pth"
    "./weights/fold_0/best.pt"
    "./weights/fold_4/best.pt"
    "./weights/fold_1/best.pt"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "${file}" ]; then
        echo "ERROR: Missing required Task 1 weight file: ${SCRIPT_DIR}/${file}"
        exit 1
    fi
done

# Remove old image if it exists
if [ -f "${SIF_NAME}" ]; then
    echo "Removing existing ${SIF_NAME}..."
    rm -f "${SIF_NAME}"
fi

apptainer build --fakeroot "${SIF_NAME}" Apptainer.def

echo ""
echo "=== Build complete: ${SCRIPT_DIR}/${SIF_NAME} ==="
