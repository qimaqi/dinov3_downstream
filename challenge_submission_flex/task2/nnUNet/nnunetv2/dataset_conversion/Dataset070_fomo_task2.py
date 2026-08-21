import shutil
from pathlib import Path

import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from tqdm import tqdm


SOURCE_ROOT = Path(
    "/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/FOMO_Tasks/Task_2/Task_2"
)


def validate_binary_segmentation(in_file: Path) -> None:
    seg = sitk.ReadImage(str(in_file))
    seg_npy = sitk.GetArrayFromImage(seg)
    unexpected_labels = sorted(set(np.unique(seg_npy).tolist()) - {0, 1})
    if unexpected_labels:
        raise RuntimeError(f"Unexpected labels in {in_file}: {unexpected_labels}. Expected only 0 and 1.")


def collect_cases(source_root: Path) -> list[tuple[str, Path, Path, Path]]:
    preprocessed_dir = source_root / "preprocessed"
    labels_dir = source_root / "labels"
    if not preprocessed_dir.is_dir():
        raise FileNotFoundError(f"Missing preprocessed directory: {preprocessed_dir}")
    if not labels_dir.is_dir():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")

    cases = []
    missing = []
    for subject_dir in sorted(preprocessed_dir.glob("sub-*")):
        session_dir = subject_dir / "ses-01"
        case_id = subject_dir.name
        flair = session_dir / "flair.nii.gz"
        dwi = session_dir / "dwi_b1000.nii.gz"
        seg = labels_dir / case_id / "ses-01" / "seg.nii.gz"
        required_files = (flair, dwi, seg)
        missing_files = [str(f) for f in required_files if not f.is_file()]
        if missing_files:
            missing.append((case_id, missing_files))
            continue
        cases.append((case_id, flair, dwi, seg))

    if missing:
        msg = "\n".join(f"{case_id}: {files}" for case_id, files in missing)
        raise FileNotFoundError(f"Missing required files for FOMO Task 2 cases:\n{msg}")
    if not cases:
        raise RuntimeError(f"No cases found under {preprocessed_dir}")

    return cases


if __name__ == "__main__":
    if nnUNet_raw is None:
        raise RuntimeError(
            "nnUNet_raw is not set. Please export nnUNet_raw before running this converter, "
            "for example to your nnU-Net raw data directory."
        )

    task_id = 70
    task_name = "FOMO26_Meningioma"
    foldername = f"Dataset{task_id:03d}_{task_name}"

    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    cases = collect_cases(SOURCE_ROOT)

    print(f"Converting {len(cases)} FOMO Task 2 cases to {out_base}")
    print("Using modalities: channel 0 = flair, channel 1 = dwi_b1000. Ignoring swi/t2s.")
    for case_id, flair, dwi, seg in tqdm(cases):
        validate_binary_segmentation(seg)
        shutil.copy(str(flair), join(imagestr, f"{case_id}_0000.nii.gz"))
        shutil.copy(str(dwi), join(imagestr, f"{case_id}_0001.nii.gz"))
        shutil.copy(str(seg), join(labelstr, f"{case_id}.nii.gz"))

    generate_dataset_json(
        out_base,
        channel_names={0: "FLAIR", 1: "DWI_b1000"},
        labels={"background": 0, "meningioma": 1},
        num_training_cases=len(cases),
        file_ending=".nii.gz",
        dataset_name=task_name,
        reference="FOMO26 Task 2",
        release="1.0",
        license="see source dataset",
        description=(
            "FOMO26 Task 2 meningioma segmentation. Converted following the asparagus preprocessing setup: "
            "use flair and dwi_b1000 only, ignore swi/t2s."
        ),
    )
