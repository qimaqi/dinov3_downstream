import shutil
from pathlib import Path

import numpy as np
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw
from tqdm import tqdm


SOURCE_ROOT = Path(
    "/usr/bmicnas02/data-biwi-01/bmicdatasets-originals/Originals/Challenge_Datasets/FOMO_Tasks/Task_4"
)
EXPECTED_LABELS = {0, 1, 2}


def resolve_source_root(source_root: Path) -> Path:
    """Accept either the public Task_4 folder or the nested Task_4/Task_4 folder."""
    if (source_root / "preprocessed").is_dir() and (source_root / "labels").is_dir():
        return source_root

    nested_root = source_root / "Task_4"
    if (nested_root / "preprocessed").is_dir() and (nested_root / "labels").is_dir():
        return nested_root

    raise FileNotFoundError(
        f"Could not find Task 4 preprocessed/labels directories under {source_root} "
        f"or {nested_root}"
    )


def _unique_labels(in_file: Path) -> set[int]:
    try:
        import SimpleITK as sitk

        seg = sitk.ReadImage(str(in_file))
        seg_npy = sitk.GetArrayFromImage(seg)
    except ImportError:
        try:
            import nibabel as nib
        except ImportError as e:
            raise RuntimeError(
                "Label validation requires SimpleITK or nibabel. Install one of them in the "
                "environment used to run this converter."
            ) from e

        seg_npy = np.asanyarray(nib.load(str(in_file)).dataobj)

    return set(np.unique(seg_npy).astype(int).tolist())


def validate_segmentation(in_file: Path) -> None:
    unexpected_labels = sorted(_unique_labels(in_file) - EXPECTED_LABELS)
    if unexpected_labels:
        raise RuntimeError(
            f"Unexpected labels in {in_file}: {unexpected_labels}. "
            f"Expected only {sorted(EXPECTED_LABELS)}."
        )


def collect_cases(source_root: Path) -> list[tuple[str, Path, Path]]:
    source_root = resolve_source_root(source_root)
    preprocessed_dir = source_root / "preprocessed"
    labels_dir = source_root / "labels"

    cases = []
    missing = []
    for t2w in sorted(preprocessed_dir.glob("sub-*/ses-*/t2w.nii.gz")):
        subject_id = t2w.parents[1].name
        session_id = t2w.parent.name
        case_id = subject_id if session_id == "ses-01" else f"{subject_id}_{session_id}"
        seg = labels_dir / subject_id / session_id / "seg.nii.gz"
        missing_files = [str(f) for f in (t2w, seg) if not f.is_file()]
        if missing_files:
            missing.append((case_id, missing_files))
            continue
        cases.append((case_id, t2w, seg))

    if missing:
        msg = "\n".join(f"{case_id}: {files}" for case_id, files in missing)
        raise FileNotFoundError(f"Missing required files for FOMO Task 4 cases:\n{msg}")
    if not cases:
        raise RuntimeError(f"No cases found under {preprocessed_dir}")

    return cases


if __name__ == "__main__":
    if nnUNet_raw is None:
        raise RuntimeError(
            "nnUNet_raw is not set. Please export nnUNet_raw before running this converter, "
            "for example to your nnU-Net raw data directory."
        )

    task_id = 71
    task_name = "FOMO26_TrigeminalNeuralgia"
    foldername = f"Dataset{task_id:03d}_{task_name}"

    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(labelstr)

    cases = collect_cases(SOURCE_ROOT)

    print(f"Converting {len(cases)} FOMO Task 4 cases to {out_base}")
    print("Using modalities: channel 0 = t2w.")
    for case_id, t2w, seg in tqdm(cases):
        validate_segmentation(seg)
        shutil.copy(str(t2w), join(imagestr, f"{case_id}_0000.nii.gz"))
        shutil.copy(str(seg), join(labelstr, f"{case_id}.nii.gz"))

    generate_dataset_json(
        out_base,
        channel_names={0: "T2w"},
        labels={"background": 0, "label1": 1, "label2": 2},
        num_training_cases=len(cases),
        file_ending=".nii.gz",
        dataset_name=task_name,
        reference="FOMO26 Task 4",
        release="1.0",
        license="see source dataset",
        description=(
            "FOMO26 Task 4 trigeminal neuralgia segmentation. Converted following the asparagus "
            "preprocessing setup: use t2w as the single input modality and the mirrored seg.nii.gz labels."
        ),
    )
