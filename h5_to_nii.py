import argparse
from dataclasses import dataclass
import h5py
import nibabel as nib
import numpy as np
import torch
import pathlib
from tqdm import tqdm


def recon(args):
    files = pathlib.Path(args.data).glob("*.h5")
    for file in tqdm(sorted(files)):
        data = np.array(h5py.File(file)["reconstruction"])
        if data.ndim > 3:
            data = data.squeeze()

        # Hardcoded for the tecfidera data
        data = np.abs(data)
        if "transverse" in str(file).split("/")[-1].split(".")[0]:
            data = np.transpose(np.flip(data, axis=(0, 2)), axes=(2, 1, 0))
        elif "coronal" in str(file).split("/")[-1].split(".")[0]:
            data = np.transpose(data, axes=(2, 0, 1))
            data = np.flip(data, axis=(0, -1))
        elif "sagittal" in str(file).split("/")[-1].split(".")[0]:
            data = np.transpose(data, axes=(0, 2, 1))
            data = np.flip(data, axis=(0, 1, 2))

        # Hardcoded the pixel size, as either 1.12 mm or 1 mm
        img = nib.Nifti1Image(data, np.eye(4) * 1.12)
        filename = "FLAIR"

        folder = pathlib.Path(
            f"{str(args.output)}/" + str(file).split("/")[-1].split(".")[0]
        )

        folder.mkdir(exist_ok=True, parents=True)

        nib.save(img, pathlib.Path(folder, f"{filename}.nii.gz"))


def seg(args):
    files = pathlib.Path(args.data).glob("*.h5")
    class_labels = {0: "background", 1: "graymatter", 2: "whitematter", 3: "lesion"}
    for file in tqdm(sorted(files)):
        data = np.squeeze(np.array(h5py.File(file)["segmentation"]))
        if data.ndim != 4 or data.shape[1] != 4:
            raise ValueError(f"{str(file).split('/')[-1]}: {data.shape}")
        if np.max(data) > 1.0:
            data = torch.softmax(torch.from_numpy(data), dim=1).numpy()

        for class_num in range(data.shape[1]):
            # Hardcoded for the tecfidera data
            seg = data[:, class_num, ...]
            if "transverse" in str(file).split("/")[-1].split(".")[0]:
                seg = np.transpose(np.flip(seg, axis=(0, 2)), axes=(2, 1, 0))
            elif "coronal" in str(file).split("/")[-1].split(".")[0]:
                seg = np.transpose(seg, axes=(2, 0, 1))
                seg = np.flip(seg, axis=(0, -1))
            elif "sagittal" in str(file).split("/")[-1].split(".")[0]:
                seg = np.transpose(seg, axes=(0, 2, 1))
                seg = np.flip(seg, axis=(0, 1, 2))

            # Hardcoded the pixel size, as either 1.12 mm or 1 mm
            img = nib.Nifti1Image(seg, np.eye(4) * 1.12)

            filename = f"FLAIR_{class_labels[class_num]}"

            folder = pathlib.Path(
                f"{str(args.output)}/" + str(file).split("/")[-1].split(".")[0]
            )

            folder.mkdir(exist_ok=True, parents=True)

            nib.save(img, pathlib.Path(folder, f"{filename}.nii.gz"))


def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["recon", "seg"], type=str, help="h5 datapath")
    parser.add_argument("data", type=pathlib.Path, help="h5 datapath")
    parser.add_argument("output", type=pathlib.Path, help="output folder")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.mode == "recon":
        recon(args)
    elif args.mode == "seg":
        seg(args)
