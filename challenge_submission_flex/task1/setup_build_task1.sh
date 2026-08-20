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

# Remove old image if it exists
if [ -f "${SIF_NAME}" ]; then
    echo "Removing existing ${SIF_NAME}..."
    rm -f "${SIF_NAME}"
fi

apptainer build --fakeroot "${SIF_NAME}" Apptainer.def

echo ""
echo "=== Build complete: ${SCRIPT_DIR}/${SIF_NAME} ==="
