#!/usr/bin/env bash
# ============================================================================
# FOMO26 Submission Build Setup
# ============================================================================
# This script prepares the build directories by symlinking shared source code
# and creating placeholder weight directories.
#
# Usage:
#   1. Run this script once: bash setup_build.sh
#   2. Copy your trained model weights into each task's weights/ directory
#   3. Build containers: cd task1 && apptainer build --fakeroot task1.sif Apptainer.def --arch amd64
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DINOV3_SRC="/home/q_ma/workspace/fomo/dinov3_downstream"

echo "=== FOMO26 Submission Build Setup ==="
echo ""

# For each task, symlink the source code and create weights directory
for task_dir in "${SCRIPT_DIR}"/task{1,2,3,4,5,6}; do
    task_name=$(basename "$task_dir")
    echo "Setting up ${task_name}..."

    # Symlink dinov3_downstream source
    if [ ! -e "${task_dir}/dinov3_downstream" ]; then
        ln -sf "$DINOV3_SRC" "${task_dir}/dinov3_downstream"
    fi

    # Create weights directory
    mkdir -p "${task_dir}/weights"

    echo "  ✓ ${task_name} ready"
done

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo ""
echo "  1. Copy your model weights into each task's weights/ directory:"
echo ""
echo "     Task 1 (Infarct):         task1/weights/best.pt + 2D_final_model.pth"
echo "     Task 2 (Meningioma):      task2/weights/nnunet_model/ + 2D_final_model.pth + 3D_final_model.pth"
echo "     Task 3 (Brain Age):       task3/weights/best.pt + 2D_final_model.pth"
echo "     Task 4 (Trigeminal):      task4/weights/nnunet_model/ + 2D_final_model.pth + 3D_final_model.pth"
echo "     Task 5 (Polymicrogyria):  task5/weights/best.pt + 2D_final_model.pth"
echo "     Task 6 (Embedding):       task6/weights/best.pt + 2D_final_model.pth"
echo ""
echo "  2. Build each container:"
echo "     cd task1 && apptainer build --fakeroot task1_infarct.sif Apptainer.def --arch amd64"
echo ""
echo "  3. Validate with: https://github.com/fomo26/container-validator"
echo ""
