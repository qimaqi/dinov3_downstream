#!/usr/bin/env bash

export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS-}"
source /usr/bmicnas03/data-biwi-01/qimaqi_data/data/miniconda3/etc/profile.d/conda.sh
conda activate flexict
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRAIN_SCRIPT="${REPO_ROOT}/downstream/3d_classify/run_fomo_cls002_infarct_patch_cls.sh"

INPUT_ROOT_DEFAULT="/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/FOMO_Tasks/Task_1/Task_1/preprocessed"
LABEL_ROOT_DEFAULT="/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/FOMO_Tasks/Task_1/Task_1/labels"
OUTPUT_ROOT_DEFAULT="${REPO_ROOT}/results/debug/task1_flex"
RESULTS_ROOT_DEFAULT="${REPO_ROOT}/results/3d_classify/fomo_slice_cls_each_robust_zscore/CLS002_FOMO26_Infarct"
BACKBONE_DEFAULT="${REPO_ROOT}/ckpts/pretrain_fomo_100k_pretrained_flexcit_base_g8_e200_p8_mri_gram/2D_final_model_fomo100k_gram.pth"
SELECTED_FOLDS_DEFAULT="fold_0,fold_2,fold_4"

INPUT_ROOT="${INPUT_ROOT_DEFAULT}"
LABEL_ROOT="${LABEL_ROOT_DEFAULT}"
OUTPUT_ROOT="${OUTPUT_ROOT_DEFAULT}"
RESULTS_ROOT="${RESULTS_ROOT_DEFAULT}"
BACKBONE_CHECKPOINT="${BACKBONE_DEFAULT}"
SELECTED_FOLDS="${SELECTED_FOLDS_DEFAULT}"
MODALITIES="flair,adc,dwi,t2s,swi"
THRESHOLD="0.4"

usage() {
    cat <<EOF
Usage: $(basename "$0") [options]

Runs Task 1 prediction over all cases in the preprocessed dataset, saves per-case
predictions, and compares them against label.txt when available.

Options:
  --input-root PATH     Input case root. Default: ${INPUT_ROOT_DEFAULT}
  --label-root PATH     Label root. Default: ${LABEL_ROOT_DEFAULT}
  --output-root PATH    Output root. Default: ${OUTPUT_ROOT_DEFAULT}
  --results-root PATH   Ensemble checkpoint root. Default: ${RESULTS_ROOT_DEFAULT}
  --backbone PATH       FlexiCT 2D checkpoint. Default: ${BACKBONE_DEFAULT}
  --selected-folds CSV  Ensemble folds. Default: ${SELECTED_FOLDS_DEFAULT}
  --modalities LIST     Comma-separated modalities to use.
                        Supported: flair,adc,dwi,t2s,swi
                        Default: ${MODALITIES}
  --threshold FLOAT     Classification threshold for summary stats. Default: ${THRESHOLD}
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input-root)
            INPUT_ROOT="$2"
            shift 2
            ;;
        --label-root)
            LABEL_ROOT="$2"
            shift 2
            ;;
        --output-root)
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --results-root)
            RESULTS_ROOT="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE_CHECKPOINT="$2"
            shift 2
            ;;
        --selected-folds)
            SELECTED_FOLDS="$2"
            shift 2
            ;;
        --modalities)
            MODALITIES="$2"
            shift 2
            ;;
        --threshold)
            THRESHOLD="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

IFS=',' read -r -a REQUESTED_MODALITIES <<< "${MODALITIES}"
if [[ ${#REQUESTED_MODALITIES[@]} -eq 0 ]]; then
    echo "ERROR: No modalities requested." >&2
    exit 1
fi

for modality in "${REQUESTED_MODALITIES[@]}"; do
    case "${modality}" in
        flair|adc|dwi|t2s|swi)
            ;;
        *)
            echo "ERROR: Unsupported modality '${modality}'." >&2
            exit 1
            ;;
    esac
done

RUN_TAG="$(echo "${MODALITIES}" | tr ',' '_' | tr -cd '[:alnum:]_')"
RUN_DIR="${OUTPUT_ROOT}/${RUN_TAG}"
CASE_OUTPUT_DIR="${RUN_DIR}/case_outputs"
SUMMARY_CSV="${RUN_DIR}/summary.csv"
LOG_FILE="${RUN_DIR}/run.log"

mkdir -p "${CASE_OUTPUT_DIR}"

exec > >(tee "${LOG_FILE}") 2>&1

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
    echo "ERROR: Training launcher not found: ${TRAIN_SCRIPT}" >&2
    exit 1
fi

if [[ ! -f "${BACKBONE_CHECKPOINT}" ]]; then
    echo "ERROR: Missing backbone checkpoint: ${BACKBONE_CHECKPOINT}" >&2
    exit 1
fi

IFS=',' read -r -a SELECTED_FOLD_ARRAY <<< "${SELECTED_FOLDS}"
if [[ ${#SELECTED_FOLD_ARRAY[@]} -eq 0 ]]; then
    echo "ERROR: No selected folds configured." >&2
    exit 1
fi
for fold in "${SELECTED_FOLD_ARRAY[@]}"; do
    if [[ ! -f "${RESULTS_ROOT}/${fold}/best.pt" ]]; then
        echo "ERROR: Missing ensemble checkpoint: ${RESULTS_ROOT}/${fold}/best.pt" >&2
        exit 1
    fi
done

export FLEXICT_2D_CHECKPOINT="${BACKBONE_CHECKPOINT}"
export TASK1_MODEL_ROOT="${RESULTS_ROOT}"
export TASK1_SELECTED_FOLDS="${SELECTED_FOLDS}"

echo "Input root: ${INPUT_ROOT}"
echo "Label root: ${LABEL_ROOT}"
echo "Output root: ${RUN_DIR}"
echo "Training launcher: ${TRAIN_SCRIPT}"
echo "Results root: ${RESULTS_ROOT}"
echo "Backbone checkpoint: ${FLEXICT_2D_CHECKPOINT}"
echo "Selected folds: ${TASK1_SELECTED_FOLDS}"
echo "Modalities: ${MODALITIES}"
echo "Threshold: ${THRESHOLD}"
echo "Preprocessing source: checkpoint-saved training args from ${TRAIN_SCRIPT}"

mapfile -t CASE_DIRS < <(find "${INPUT_ROOT}" -mindepth 2 -maxdepth 2 -type d | sort)
if [[ ${#CASE_DIRS[@]} -eq 0 ]]; then
    echo "ERROR: No case directories found in ${INPUT_ROOT}" >&2
    exit 1
fi

printf "case_id,session,modalities_used,probability,pred_label,gt_label,correct,output_file\n" > "${SUMMARY_CSV}"

TOTAL=0
NUM_WITH_LABEL=0
NUM_CORRECT=0

for case_dir in "${CASE_DIRS[@]}"; do
    subject="$(basename "$(dirname "${case_dir}")")"
    session="$(basename "${case_dir}")"
    case_id="${subject}_${session}"
    out_file="${CASE_OUTPUT_DIR}/${case_id}.txt"

    cmd=(bash "${SCRIPT_DIR}/local_debug_before_apptainer_bmic_flex.sh" --output "${out_file}")
    used_modalities=()

    for modality in "${REQUESTED_MODALITIES[@]}"; do
        case "${modality}" in
            flair)
                path="${case_dir}/flair.nii.gz"
                arg="--flair"
                ;;
            adc)
                path="${case_dir}/adc.nii.gz"
                arg="--adc"
                ;;
            dwi)
                path="${case_dir}/dwi_b1000.nii.gz"
                arg="--dwi"
                ;;
            t2s)
                path="${case_dir}/t2s.nii.gz"
                arg="--t2s"
                ;;
            swi)
                path="${case_dir}/swi.nii.gz"
                arg="--swi"
                ;;
        esac

        if [[ -f "${path}" ]]; then
            cmd+=("${arg}" "${path}")
            used_modalities+=("${modality}")
        fi
    done

    if [[ ${#used_modalities[@]} -eq 0 ]]; then
        echo "Skipping ${case_id}: none of the requested modalities are present."
        continue
    fi

    echo ""
    echo "=== Running ${case_id} with modalities: $(IFS=,; echo "${used_modalities[*]}") ==="
    "${cmd[@]}"

    probability="$(tr -d '[:space:]' < "${out_file}")"
    pred_label="$(python3 - <<PY
prob = float("${probability}")
threshold = float("${THRESHOLD}")
print(1 if prob >= threshold else 0)
PY
)"

    gt_label=""
    correct=""
    label_file="${LABEL_ROOT}/${subject}/${session}/label.txt"
    if [[ -f "${label_file}" ]]; then
        gt_label="$(tr -d '[:space:]' < "${label_file}")"
        correct="$(python3 - <<PY
pred = int("${pred_label}")
gt = int(float("${gt_label}"))
print(1 if pred == gt else 0)
PY
)"
        NUM_WITH_LABEL=$((NUM_WITH_LABEL + 1))
        NUM_CORRECT=$((NUM_CORRECT + correct))
    fi

    printf "%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "${subject}" \
        "${session}" \
        "$(IFS=,; echo "${used_modalities[*]}")" \
        "${probability}" \
        "${pred_label}" \
        "${gt_label}" \
        "${correct}" \
        "${out_file}" >> "${SUMMARY_CSV}"

    TOTAL=$((TOTAL + 1))
done

echo ""
echo "=== Finished ==="
echo "Cases processed: ${TOTAL}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "Per-case outputs: ${CASE_OUTPUT_DIR}"

if [[ ${NUM_WITH_LABEL} -gt 0 ]]; then
    accuracy="$(python3 - <<PY
num_correct = ${NUM_CORRECT}
num_with_label = ${NUM_WITH_LABEL}
print(f"{num_correct / num_with_label:.4f}")
PY
)"
    echo "Labeled cases: ${NUM_WITH_LABEL}"
    echo "Accuracy @ threshold ${THRESHOLD}: ${accuracy}"
fi
