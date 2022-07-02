# encoding: utf-8
__author__ = "Dimitrios Karkalousos"

import argparse
import json
import os
import h5py
import numpy as np
import sys
import time
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from mridc.collections.common.parts.fft import fft2c, ifft2c


def main(args):
    init_start = time.perf_counter()

    with open(
        "/data/projects/recon/data/public/qdess/v1-release/annotations/v1.0.0/test.json"
    ) as f:
        config = json.load(f)
    files = [os.path.join(args.file_path, i["file_name"]) for i in config["images"]]

    print("Saving data to h5 format...")
    start_time = time.perf_counter()
    for file in tqdm((files)):

        data = h5py.File(str(file), "r")
        kspace = data["kspace"][:, :, :, 0, :]  # 1st Echo

        # kspace = np.fft.fftshift(kspace, axes=(1, 2))  # shift mask must be on
        # imspace = np.fft.ifftn(kspace, axes=(1, 2))
        # imspace = np.fft.ifftshift(imspace, axes=(1, 2))
        # imspace = imspace / np.max(np.abs(imspace))

        kspace = torch.from_numpy(kspace)
        imspace = torch.view_as_complex(
            ifft2c(torch.view_as_real(kspace), fft_dim=[1, 2])
        ).numpy()
        imspace = imspace / np.max(np.abs(imspace))
        kspace = torch.view_as_complex(
            fft2c(torch.view_as_real(torch.from_numpy(imspace)), fft_dim=[1, 2])
        ).numpy()

        csm = data["maps"][..., 0]  # 1st Echo
        csm = csm / np.sum(np.abs(csm), -1, keepdims=True)
        csm = csm / np.max(np.abs(csm))

        target = np.sum(imspace * csm.conj(), -1)
        # target = data["target"][:, :, :, 0, 0]
        target = target / np.max(np.abs(target))

        kspace = np.transpose(kspace, (0, 3, 1, 2))
        csm = np.transpose(csm, (0, 3, 1, 2))

        masks_out_dir = Path("/".join(str(args.file_path).split("/")[:-2]) + "/masks/")
        masks_out_dir.mkdir(parents=True, exist_ok=True)

        masks = dict(data["masks"]["/masks"])
        new_masks = {}
        for acc, mask in masks.items():
            # readout always oversampled to 512 from 416, so first padding size is fixed to (48, 48)
            padding_size = (kspace.shape[2] - mask.shape[1]) // 2
            mask_padded = np.pad(mask, ((48, 48), (padding_size, padding_size)))
            np.save(f"{str(masks_out_dir)}/{acc}.npy", mask_padded)

        out_dir = Path("/".join(str(args.file_path).split("/")[:-2]) + "/h5/")
        out_dir.mkdir(parents=True, exist_ok=True)

        hf = h5py.File(Path(f"{str(out_dir)}/{str(file).split('/')[-1]}"), "w")
        hf.create_dataset("kspace", data=kspace.astype(np.complex64))
        hf.create_dataset("sensitivity_map", data=csm.astype(np.complex64))
        hf.create_dataset("target", data=np.abs(target).astype(np.float32))
        hf.close()

    print(
        "Done! Data saved into h5 format. It took",
        time.perf_counter() - start_time,
        "s \n",
    )


# noinspection PyTypeChecker
def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path", type=str, help="Path of the files to be converted."
    )
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args(sys.argv[1:])
    args.out_dir = "/".join(args.file_path.split("/")[:-2])
    main(args)
