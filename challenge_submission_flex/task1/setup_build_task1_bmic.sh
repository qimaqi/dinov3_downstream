#!/usr/bin/env bash
# Build the Apptainer container for Task 1 (Infarct Detection)
# Usage: bash setup_build_task1.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

SIF_NAME="task1_infarct.sif"

echo "=== Building Task 1 container ==="
echo "  Working dir: ${SCRIPT_DIR}"
echo "  Definition:  Apptainer.def"
echo "  Output:      ${SIF_NAME}"
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

# Try building without fakeroot.
# Strategy: build a sandbox (unprivileged), then convert to SIF.
# This avoids requiring fakeroot or real root privileges.
SANDBOX_DIR="${SIF_NAME%.sif}_sandbox"

echo "Step 1/2: Building sandbox (unprivileged)..."
rm -rf "${SANDBOX_DIR}"
apptainer build --fix-perms "${SANDBOX_DIR}" Apptainer.def

echo "Step 2/2: Converting sandbox to SIF..."
apptainer build "${SIF_NAME}" "${SANDBOX_DIR}"

# Clean up sandbox
rm -rf "${SANDBOX_DIR}"

echo ""
echo "=== Build complete: ${SCRIPT_DIR}/${SIF_NAME} ==="
