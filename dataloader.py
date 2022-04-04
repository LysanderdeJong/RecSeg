import os

import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader

import pytorch_lightning as pl

from fastmri.data.transforms import to_tensor
from fastmri import rss

from einops import rearrange

import h5py
import pickle
import nibabel as nib
import json
from pathlib import Path


class skmtea(Dataset):
    def __init__(
        self,
        split,
        data_files,
        data_root="/data/projects/recon/data/public/qdess/v1-release/",
        mri_data_path="files_recon_calib-24/",
        segmentation_path="segmentation_masks/raw-data-track/",
        seq_len=1,
        compact_masks=False,
        use_cache=True,
    ):
        self.split = split
        self.data_root = data_root
        self.mri_data_path = mri_data_path
        self.segmentation_path = segmentation_path
        self.seq_len = seq_len
        self.compact_masks = compact_masks

        cache_file = f"./.cache/skm_cache_{split}.pkl"
        if os.path.isfile(cache_file) and use_cache:
            with open(cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
            print("Using saved cache.")
        else:
            dataset_cache = []

        file_names = data_files[split]
        file_names.sort()

        if os.path.isfile(cache_file) and use_cache:
            self.mri_slices = dataset_cache[0]
            self.mask_slices = dataset_cache[1]
        else:
            self.mri_slices = []
            self.mask_slices = []

            print("Generating cache file.")

            for file_name in file_names:
                mri_path = os.path.join(data_root, mri_data_path, file_name + ".h5")
                mask_path = os.path.join(
                    data_root, segmentation_path, file_name + ".nii.gz"
                )

                mri = h5py.File(mri_path, "r")["target"].shape[2]
                mask = nib.load(mask_path).dataobj.shape[2]

                mri_num_slices = mri - self.seq_len + 1
                mask_num_slices = mask - self.seq_len + 1

                self.mri_slices += [(mri_path, i) for i in range(mri_num_slices)]
                self.mask_slices += [(mask_path, i) for i in range(mask_num_slices)]

            assert len(self.mri_slices) == len(self.mask_slices)

            dataset_cache.append(self.mri_slices)
            dataset_cache.append(self.mask_slices)

            if use_cache:
                os.makedirs("./.cache", exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
                print(f"Saving cache to {cache_file}.")
                del dataset_cache

    def __len__(self):
        return len(self.mask_slices)

    def __getitem__(self, idx):
        fmri, mri_slice = self.mri_slices[idx]
        fmask, mask_slice = self.mask_slices[idx]

        mri = h5py.File(fmri, "r", libver="latest")["target"][
            :, :, mri_slice : mri_slice + self.seq_len, :, :
        ]
        mask = np.array(
            nib.load(fmask).dataobj[:, :, mask_slice : mask_slice + self.seq_len]
        )

        mri_image = rss(to_tensor(mri), dim=-3)
        seg_mask = self.convert_mask(mask, compact=self.compact_masks)

        mri_image = rearrange(mri_image, "x y z () i -> z i x y")
        seg_mask = rearrange(seg_mask, "h w c s -> c s h w")

        return mri_image, seg_mask

    def convert_mask(self, mask, n_classes=7, compact=False):
        x, y, z = mask.shape
        out = np.zeros((x, y, z, n_classes), dtype=np.bool)
        for i in range(n_classes):
            out[:, :, :, i] = mask == i
        if compact:
            out[:, :, :, 3] = np.logical_or(out[:, :, :, 3], out[:, :, :, 4])
            out[:, :, :, 4] = np.logical_or(out[:, :, :, 5], out[:, :, :, 6])
            out = out[:, :, :, :5]
        return out.astype(np.uint8)


class SKMDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        mri_data_path,
        segmentation_path,
        annotation_path=None,
        seq_len=1,
        use_cache=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.file_split = {}

        if annotation_path is None:
            file_names = os.listdir(os.path.join(data_root, mri_data_path))
            file_names = [i[:-3] for i in file_names]
            self.file_split["all"] = file_names
        else:
            data_splits = os.listdir(os.path.join(data_root, annotation_path))
            data_splits = [i[:-5] for i in data_splits]
            for split in data_splits:
                with open(
                    os.path.join(data_root, annotation_path, split + ".json")
                ) as f:
                    config = json.load(f)
                    self.file_split[split] = [
                        i["file_name"][:-3] for i in config["images"]
                    ]

    def _make_dataset(self, split, data_files):
        return skmtea(
            split,
            data_files,
            self.hparams.data_root,
            self.hparams.mri_data_path,
            self.hparams.segmentation_path,
            self.hparams.seq_len,
            self.hparams.compact_masks,
            self.hparams.use_cache,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.hparams.annotation_path:
                self.train = self._make_dataset("train", self.file_split)
                self.val = self._make_dataset("val", self.file_split)
            else:
                dataset = self._make_dataset("all", self.file_split)
                self.train, self.val = random_split(
                    dataset,
                    [
                        int(self.hparams.train_fraction * len(dataset)),
                        len(dataset)
                        - (int(self.hparams.train_fraction * len(dataset))),
                    ],
                )
                self.val, self.test = random_split(
                    self.val, [len(self.val) // 2, len(self.val) - len(self.val) // 2]
                )

        if stage == "test" or stage is None:
            if self.hparams.annotation_path:
                self.test = self._make_dataset("test", self.file_split)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")

        # dataset arguments
        parser.add_argument(
            "--data_root",
            default="/data/projects/recon/data/public/qdess/v1-release/",
            type=str,
            help="Path to the data root",
        )
        parser.add_argument(
            "--mri_data_path",
            default="files_recon_calib-24/",
            type=str,
            help="Path to the raw mri data from the root.",
        )
        parser.add_argument(
            "--segmentation_path",
            default="segmentation_masks/raw-data-track/",
            type=str,
            help="Path to the segmentation maps from the root.",
        )
        parser.add_argument(
            "--annotation_path",
            default="annotations/v1.0.0/",
            type=str,
            help="Path to the annotation maps from the root. Optional",
        )
        parser.add_argument(
            "--seq_len", default=1, type=int, help="Size of the slice looked at."
        )
        parser.add_argument(
            "--train_fraction",
            default=0.8,
            type=float,
            help="Fraction of th data used for training.",
        )
        parser.add_argument(
            "--use_cache",
            default=True,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )
        parser.add_argument(
            "--compact_masks",
            default=False,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parent_parser


class brain_dwi(Dataset):
    def __init__(
        self,
        split,
        data_files,
        data_root="/data/projects/dwi_aisd/",
        mri_data_path="DWIs_nii/",
        segmentation_path="masks_DWI/",
        seq_len=1,
        use_cache=True,
    ):
        self.split = split
        self.data_root = data_root
        self.mri_data_path = mri_data_path
        self.segmentation_path = segmentation_path
        self.seq_len = seq_len
        self.input_transform = transforms.Resize([256,], antialias=True,)
        self.target_transform = transforms.Resize(
            [256,], interpolation=transforms.InterpolationMode.NEAREST,
        )

        cache_file = f"./.cache/dwi_cache_{split}.pkl"
        if os.path.isfile(cache_file) and use_cache:
            with open(cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
            print("Using saved cache.")
        else:
            dataset_cache = []

        file_names = data_files[split]
        file_names.sort()

        if os.path.isfile(cache_file) and use_cache:
            self.mri_slices = dataset_cache[0]
            self.mask_slices = dataset_cache[1]
        else:
            self.mri_slices = []
            self.mask_slices = []

            print("Generating cache file.")

            for file_name in file_names:
                mri_path = os.path.join(
                    data_root, mri_data_path, file_name + "DWI.nii.gz"
                )
                mask_path = os.path.join(
                    data_root, segmentation_path, file_name + "mask.nii.gz"
                )

                mri = nib.load(mri_path).dataobj.shape[2]
                mask = nib.load(mask_path).dataobj.shape[2]

                mri_num_slices = mri - self.seq_len + 1
                mask_num_slices = mask - self.seq_len + 1

                self.mri_slices += [(mri_path, i) for i in range(mri_num_slices)]
                self.mask_slices += [(mask_path, i) for i in range(mask_num_slices)]

            assert len(self.mri_slices) == len(self.mask_slices)

            dataset_cache.append(self.mri_slices)
            dataset_cache.append(self.mask_slices)

            if use_cache:
                os.makedirs("./.cache", exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
                print(f"Saving cache to {cache_file}.")
                del dataset_cache

    def __len__(self):
        return len(self.mask_slices)

    def __getitem__(self, idx):
        fmri, mri_slice = self.mri_slices[idx]
        fmask, mask_slice = self.mask_slices[idx]

        mri = np.array(
            nib.as_closest_canonical(nib.load(fmri)).dataobj[
                :, :, mri_slice : mri_slice + self.seq_len
            ]
        )
        mask = np.array(
            nib.as_closest_canonical(nib.load(fmask)).dataobj[
                :, :, mask_slice : mask_slice + self.seq_len
            ]
        )

        mri_image = torch.from_numpy(mri.astype(np.float32))
        if len(mri_image.shape) == 4:
            mri_image = mri_image[:, :, :, 0]

        seg_mask = torch.from_numpy(self.convert_mask(mask, compact=True))

        mri_image = rearrange(mri_image, "x y z -> z () x y")
        seg_mask = rearrange(seg_mask, "h w c s -> c s h w")

        if self.input_transform:
            mri_image = self.input_transform(mri_image)

        if self.target_transform:
            seg_mask = self.target_transform(seg_mask)

        return mri_image, seg_mask

    def convert_mask(self, mask, n_classes=6, compact=False):
        x, y, z = mask.shape
        out = np.zeros((x, y, z, n_classes), dtype=np.bool)
        for i in range(n_classes):
            out[:, :, :, i] = mask == i
        if compact:
            out[:, :, :, 1] = np.logical_or.reduce(
                (
                    out[:, :, :, 1],
                    out[:, :, :, 2],
                    out[:, :, :, 3],
                    out[:, :, :, 4],
                    out[:, :, :, 5],
                )
            )
            out = out[:, :, :, :2]
        return out.astype(np.uint8)


class BrainDWIDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_root,
        mri_data_path,
        segmentation_path,
        seq_len=1,
        use_cache=True,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.file_split = {}

        file_names = os.listdir(os.path.join(data_root, segmentation_path))
        file_names = [i[:-11] for i in file_names]
        self.file_split["all"] = file_names

    def _make_dataset(self, split, data_files):
        return brain_dwi(
            split,
            data_files,
            self.hparams.data_root,
            self.hparams.mri_data_path,
            self.hparams.segmentation_path,
            self.hparams.seq_len,
            self.hparams.use_cache,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = self._make_dataset("all", self.file_split)
            self.train, self.val = random_split(
                dataset,
                [
                    int(self.hparams.train_fraction * len(dataset)),
                    len(dataset) - (int(self.hparams.train_fraction * len(dataset))),
                ],
            )
            self.val, self.test = random_split(
                self.val, [len(self.val) // 2, len(self.val) - len(self.val) // 2]
            )

        if stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")

        # dataset arguments
        parser.add_argument(
            "--data_root",
            default="/data/projects/dwi_aisd/",
            type=str,
            help="Path to the data root",
        )
        parser.add_argument(
            "--mri_data_path",
            default="DWIs_nii/",
            type=str,
            help="Path to the raw mri data from the root.",
        )
        parser.add_argument(
            "--segmentation_path",
            default="masks_DWI/",
            type=str,
            help="Path to the segmentation maps from the root.",
        )
        parser.add_argument(
            "--annotation_path",
            default=None,
            type=str,
            help="Path to the annotation maps from the root. Optional",
        )
        parser.add_argument(
            "--seq_len", default=1, type=int, help="Size of the slice looked at."
        )
        parser.add_argument(
            "--train_fraction",
            default=0.8,
            type=float,
            help="Fraction of th data used for training.",
        )
        parser.add_argument(
            "--use_cache",
            default=False,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )
        parser.add_argument(
            "--compact_masks",
            default=False,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parent_parser


class TecFidera(Dataset):
    def __init__(
        self,
        split,
        data_files,
        data_root="/data/projects/tecfidera/data/h5_recon_dataset/",
        seq_len=1,
        use_cache=True,
        compact_masks=True,
    ):
        self.split = split
        self.data_root = data_root
        self.seq_len = seq_len
        self.compact_masks = compact_masks
        self.input_transform = transforms.Resize([256, 256], antialias=True,)
        self.target_transform = transforms.Resize(
            [256, 256], interpolation=transforms.InterpolationMode.NEAREST,
        )

        cache_file = f"./.cache/tecfidera_cache_{split}.pkl"
        if os.path.isfile(cache_file) and use_cache:
            with open(cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
            print("Using saved cache.")
        else:
            dataset_cache = []

        file_names = data_files[split]
        file_names.sort()

        if os.path.isfile(cache_file) and use_cache:
            self.mri_slices = dataset_cache[0]
            self.mask_slices = dataset_cache[1]
        else:
            self.mri_slices = []
            self.mask_slices = []

            print("Generating cache file.")

            for file_name in file_names:
                mri_path = os.path.join(data_root, file_name)

                mri = h5py.File(mri_path, "r")

                mri_shape = mri["reconstruction_sense"].shape[0]
                mask_shape = mri["lesion_segmentation"].shape[0]

                mri_num_slices = mri_shape - self.seq_len + 1
                mask_num_slices = mask_shape - self.seq_len + 1

                self.mri_slices += [(mri_path, i) for i in range(mri_num_slices)]
                self.mask_slices += [(mri_path, i) for i in range(mask_num_slices)]

            assert len(self.mri_slices) == len(self.mask_slices)

            dataset_cache.append(self.mri_slices)
            dataset_cache.append(self.mask_slices)

            if use_cache:
                os.makedirs("./.cache", exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
                print(f"Saving cache to {cache_file}.")
                del dataset_cache

    def __len__(self):
        return len(self.mask_slices)

    def __getitem__(self, idx):
        fmri, mri_slice = self.mri_slices[idx]
        fmask, mask_slice = self.mask_slices[idx]

        # print(h5py.File(fmri, "r")["reconstruction_sense"].shape)

        mri = np.array(
            h5py.File(fmri, "r")["reconstruction_sense"][
                mri_slice : mri_slice + self.seq_len, :, :, :
            ]
        )
        mask = np.array(
            h5py.File(fmask, "r")["lesion_segmentation"][
                mask_slice : mask_slice + self.seq_len, :, :
            ]
        )

        mri_image = to_tensor(mri)
        seg_mask = torch.from_numpy(self.convert_mask(mask, compact=self.compact_masks))

        mri_image = rearrange(mri_image, "z () x y c -> z c x y")
        seg_mask = rearrange(seg_mask, "c h w s -> c s h w")

        # if self.input_transform:
        #     mri_image = self.input_transform(mri_image)

        # if self.target_transform:
        #     seg_mask = self.target_transform(seg_mask)

        return mri_image, seg_mask

    def convert_mask(self, mask, n_classes=2, compact=False):
        x, y, z = mask.shape
        out = np.zeros((x, y, z, n_classes), dtype=np.bool)
        for i in range(n_classes):
            out[:, :, :, i] = mask == i
        return out.astype(np.uint8)


class TecFideraDataModule(pl.LightningDataModule):
    def __init__(
        self, data_root, seq_len=1, use_cache=True, **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.file_split = {}

        file_names = os.listdir(os.path.join(data_root))
        self.file_split["all"] = file_names

    def _make_dataset(self, split, data_files):
        return TecFidera(
            split,
            data_files,
            self.hparams.data_root,
            self.hparams.seq_len,
            self.hparams.use_cache,
            self.hparams.compact_masks,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = self._make_dataset("all", self.file_split)
            self.train, self.val = random_split(
                dataset,
                [
                    int(self.hparams.train_fraction * len(dataset)),
                    len(dataset) - (int(self.hparams.train_fraction * len(dataset))),
                ],
            )
            self.val, self.test = random_split(
                self.val, [len(self.val) // 2, len(self.val) - len(self.val) // 2]
            )

        if stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")

        # dataset arguments
        parser.add_argument(
            "--data_root",
            default="/data/projects/tecfidera/data/h5_recon_dataset/",
            type=str,
            help="Path to the data root",
        )
        parser.add_argument(
            "--seq_len", default=1, type=int, help="Size of the slice looked at."
        )
        parser.add_argument(
            "--train_fraction",
            default=0.8,
            type=float,
            help="Fraction of th data used for training.",
        )
        parser.add_argument(
            "--use_cache",
            default=False,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )
        parser.add_argument(
            "--compact_masks",
            default=True,
            type=bool,
            help="Whether to cache dataset metadata in a pkl file",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parent_parser


from mridc.collections.reconstruction.data.mri_data import FastMRISliceDataset
from mridc.collections.reconstruction.parts.transforms import MRIDataTransforms
from mridc.collections.reconstruction.data.subsample import create_mask_for_mask_type


class MRISliceDataset(FastMRISliceDataset):
    def __init__(
        self,
        root,
        challenge: str = "multicoil",
        transform=None,
        sense_root=None,
        use_dataset_cache: bool = False,
        sample_rate: float = 1.0,
        volume_sample_rate: float = None,
        dataset_cache_file: str = "dataset_cache.yaml",
        num_cols: int = None,
        mask_root: str = None,
        mask_type: str = "gaussian2d",
        shift_mask: bool = False,
        accelerations=[4, 6, 8, 10],
        center_fractions=[0.7, 0.7, 0.7, 0.7],
        scale: float = 0.02,
        normalize_inputs: bool = True,
        crop_size=None,
        crop_before_masking: bool = True,
        kspace_zero_filling_size=None,
        fft_type: str = "orthogonal",
        use_seed: bool = True,
    ):

        if mask_type is not None and mask_type != "None":
            mask_func = (
                [
                    create_mask_for_mask_type(mask_type, [cf] * 2, [acc] * 2)
                    for acc, cf in zip(accelerations, center_fractions)
                ]
                if len(accelerations) > 2
                else [
                    create_mask_for_mask_type(
                        mask_type, center_fractions, accelerations
                    )
                ]
            )

        else:
            mask_func = None  # type: ignore
            scale = 0.02

        transform = (
            MRIDataTransforms(
                mask_func=mask_func,
                shift_mask=shift_mask,
                mask_center_scale=scale,
                normalize_inputs=normalize_inputs,
                crop_size=crop_size,
                crop_before_masking=crop_before_masking,
                kspace_zero_filling_size=kspace_zero_filling_size,
                fft_type=fft_type,
                use_seed=use_seed,
            )
            if transform is None
            else transform
        )

        sample_rate = 1 if sample_rate is None else sample_rate

        super().__init__(
            root=root,
            challenge=challenge,
            transform=transform,
            sense_root=sense_root,
            use_dataset_cache=use_dataset_cache,
            sample_rate=sample_rate,
            volume_sample_rate=volume_sample_rate,
            dataset_cache_file=dataset_cache_file,
            num_cols=num_cols,
            mask_root=mask_root,
        )

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.examples[i]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice].astype(np.complex64)

            if "sensitivity_map" in hf:
                sensitivity_map = hf["sensitivity_map"][dataslice].astype(np.complex64)
            elif self.sense_root is not None and self.sense_root != "None":
                with h5py.File(
                    Path(self.sense_root)
                    / Path(str(fname).split("/")[-2])
                    / fname.name,
                    "r",
                ) as sf:
                    sensitivity_map = (
                        sf["sensitivity_map"][dataslice]
                        if "sensitivity_map" in sf
                        or "sensitivity_map" in next(iter(sf.keys()))
                        else sf["sense"][dataslice]
                    )
                    sensitivity_map = sensitivity_map.squeeze().astype(np.complex64)
            else:
                sensitivity_map = np.array([])

            if "mask" in hf:
                mask = np.asarray(hf["mask"])

                if mask.ndim == 3:
                    mask = mask[dataslice]

            elif self.mask_root is not None and self.mask_root != "None":
                mask_path = Path(self.mask_root) / Path(
                    str(fname.name).split(".")[0] + ".npy"
                )
                mask = np.load(str(mask_path))
            else:
                mask = None

            eta = (
                hf["eta"][dataslice].astype(np.complex64)
                if "eta" in hf
                else np.array([])
            )

            if "reconstruction_sense" in hf:
                self.recons_key = "reconstruction_sense"

            target = hf[self.recons_key][dataslice] if self.recons_key in hf else None

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if sensitivity_map.shape != kspace.shape:
            sensitivity_map = np.transpose(sensitivity_map, (2, 0, 1))

        return (
            (kspace, sensitivity_map, mask, eta, target, attrs, fname.name, dataslice,)
            if self.transform is None
            else self.transform(
                kspace,
                sensitivity_map,
                mask,
                eta,
                target,
                attrs,
                fname.name,
                dataslice,
            )
        )


class TecFideraMRIDataModule(pl.LightningDataModule):
    def __init__(
        self, data_root, **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

    def _make_dataset(self):
        return MRISliceDataset(
            root=self.hparams.data_root,
            challenge=self.hparams.challenge,
            sample_rate=self.hparams.sample_rate,
            mask_type=self.hparams.mask_type,
            shift_mask=self.hparams.shift_mask,
            accelerations=self.hparams.accelerations,
            center_fractions=self.hparams.center_fractions,
            scale=self.hparams.mask_center_scale,
            normalize_inputs=self.hparams.normalize_inputs,
            crop_size=self.hparams.crop_size,
            crop_before_masking=self.hparams.crop_before_masking,
            kspace_zero_filling_size=self.hparams.kspace_zero_filling_size,
            fft_type=self.hparams.fft_type_data,
            use_seed=self.hparams.use_seed,
        )

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            dataset = self._make_dataset()
            self.train, self.val = random_split(
                dataset,
                [
                    int(self.hparams.train_fraction * len(dataset)),
                    len(dataset) - (int(self.hparams.train_fraction * len(dataset))),
                ],
            )
            self.val, self.test = random_split(
                self.val, [len(self.val) // 2, len(self.val) - len(self.val) // 2]
            )

        if stage == "test" or stage is None:
            pass

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.num_workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")

        # dataset arguments
        parser.add_argument(
            "--data_root",
            default="/data/projects/tecfidera/data/h5_recon_dataset/",
            type=str,
            help="Path to the data root.",
        )
        parser.add_argument(
            "--challenge",
            default="multicoil",
            type=str,
            help="Challange category taken from FastMRI",
        )
        parser.add_argument(
            "--sample_rate",
            default=1.0,
            type=float,
            help="Fraction of the data to use.",
        )
        parser.add_argument(
            "--mask_type",
            default="gaussian2d",
            type=str,
            help="The string representation of the mask type.",
        )
        parser.add_argument(
            "--shift_mask",
            action="store_true",
            default=False,
            help="The string representation of the mask type.",
        )
        parser.add_argument(
            "--accelerations",
            nargs="+",
            type=int,
            default=[4, 6, 8, 10],
            help="Mask acceleration factors",
        )
        parser.add_argument(
            "--center_fractions",
            nargs="+",
            type=float,
            default=[0.7, 0.7, 0.7, 0.7],
            help="Center fraction of the mask",
        )
        parser.add_argument(
            "--mask_center_scale", type=float, default=0.02, help="Mask center scale",
        )
        parser.add_argument(
            "--normalize_inputs",
            action="store_false",
            default=True,
            help="Whether or not to normaliz the input to the network",
        )
        parser.add_argument(
            "--crop_size",
            nargs="+",
            type=int,
            default=None,
            help="Size to crop kspace to",
        )
        parser.add_argument(
            "--crop_before_masking",
            action="store_false",
            default=True,
            help="Whether or not the cropping happens before or after the masking",
        )
        parser.add_argument(
            "--kspace_zero_filling_size",
            nargs="+",
            type=int,
            default=None,
            help="Size to kspace to be filled",
        )
        parser.add_argument(
            "--fft_type_data",
            default="backward",
            type=str,
            help="The type of fft normilization to use.",
        )
        parser.add_argument(
            "--use_seed",
            action="store_false",
            default=True,
            help="Whether to use the seed",
        )

        parser.add_argument(
            "--train_fraction",
            default=0.85,
            type=float,
            help="Fraction of th data used for training.",
        )

        # data loader arguments
        parser.add_argument(
            "--batch_size", default=1, type=int, help="Data loader batch size"
        )
        parser.add_argument(
            "--num_workers",
            default=4,
            type=int,
            help="Number of workers to use in data loader",
        )

        return parent_parser
