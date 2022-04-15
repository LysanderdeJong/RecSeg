import os

import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset


from fastmri.data.transforms import to_tensor
from fastmri import rss

from einops import rearrange

import h5py
import pickle
import nibabel as nib
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
                    data_root, segmentation_path, f"{file_name}.nii.gz"
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
                    data_root, mri_data_path, f"{file_name}DWI.nii.gz"
                )
                mask_path = os.path.join(
                    data_root, segmentation_path, f"{file_name}mask.nii.gz"
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
        accelerations=None,
        center_fractions=None,
        scale: float = 0.02,
        normalize_inputs: bool = True,
        crop_size=None,
        crop_before_masking: bool = True,
        kspace_zero_filling_size=None,
        fft_type: str = "orthogonal",
        use_seed: bool = True,
        segmentation: bool = False,
        seq_len: int = 1,
    ):

        if accelerations is None:
            accelerations = [4, 6, 8, 10]
        if center_fractions is None:
            center_fractions = [0.7, 0.7, 0.7, 0.7]
        self.segmentation = segmentation
        self.seq_len = seq_len

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

        if challenge not in ("singlecoil", "multicoil"):
            raise ValueError('challenge should be either "singlecoil" or "multicoil"')

        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError(
                "either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but not both"
            )

        self.sense_root = sense_root
        self.mask_root = mask_root

        self.dataset_cache_file = Path(dataset_cache_file)

        self.transform = transform
        self.recons_key = (
            "reconstruction_esc" if challenge == "singlecoil" else "reconstruction_rss"
        )
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if self.dataset_cache_file.exists() and use_dataset_cache:
            with open(self.dataset_cache_file, "rb") as f:
                dataset_cache = yaml.safe_load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(root) is None or not use_dataset_cache:
            files = list(Path(root).iterdir())
            for fname in sorted(files):
                metadata, num_slices = self._retrieve_metadata(fname)
                num_slices -= self.seq_len + 1
                self.examples += [
                    (fname, slice_ind, metadata) for slice_ind in range(num_slices)
                ]

            if dataset_cache.get(root) is None and use_dataset_cache:
                dataset_cache[root] = self.examples
                logging.info(f"Saving dataset cache to {self.dataset_cache_file}.")
                with open(self.dataset_cache_file, "wb") as f:  # type: ignore
                    yaml.dump(dataset_cache, f)  # type: ignore
        else:
            logging.info(f"Using dataset cache from {self.dataset_cache_file}.")
            self.examples = dataset_cache[root]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list({f[0].stem for f in self.examples}))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.examples = [
                example for example in self.examples if example[0].stem in sampled_vols
            ]

        if num_cols:
            self.examples = [ex for ex in self.examples if ex[2]["encoding_size"][1] in num_cols]  # type: ignore

    def convert_mask(self, mask, n_classes=2, compact=False):
        x, y, z = mask.shape
        out = np.zeros((x, y, z, n_classes), dtype=np.bool)
        for i in range(n_classes):
            out[:, :, :, i] = mask == i
        out = rearrange(out, "c h w s -> c s h w")
        return out.astype(np.uint8)

    def __getitem__(self, i: int):
        fname, dataslice, metadata = self.examples[i]
        with h5py.File(fname, "r") as hf:
            kspace = hf["kspace"][dataslice : dataslice + self.seq_len].astype(
                np.complex64
            )

            if "sensitivity_map" in hf:
                sensitivity_map = hf["sensitivity_map"][
                    dataslice : dataslice + self.seq_len
                ].astype(np.complex64)
            elif self.sense_root is not None and self.sense_root != "None":
                with h5py.File(
                    Path(self.sense_root)
                    / Path(str(fname).split("/")[-2])
                    / fname.name,
                    "r",
                ) as sf:
                    sensitivity_map = (
                        sf["sensitivity_map"][dataslice : dataslice + self.seq_len]
                        if "sensitivity_map" in sf
                        or "sensitivity_map" in next(iter(sf.keys()))
                        else sf["sense"][dataslice : dataslice + self.seq_len]
                    )
                    sensitivity_map = sensitivity_map.squeeze().astype(np.complex64)
            else:
                sensitivity_map = np.array([])

            if "mask" in hf:
                mask = np.asarray(hf["mask"])

                if mask.ndim == 3:
                    mask = mask[dataslice : dataslice + self.seq_len]

            elif self.mask_root is not None and self.mask_root != "None":
                mask_path = Path(self.mask_root) / Path(
                    str(fname.name).split(".")[0] + ".npy"
                )
                mask = np.load(str(mask_path))
            else:
                mask = None

            eta = (
                hf["eta"][dataslice : dataslice + self.seq_len].astype(np.complex64)
                if "eta" in hf
                else np.array([])
            )

            if "reconstruction_sense" in hf:
                self.recons_key = "reconstruction_sense"

            target = (
                hf[self.recons_key][dataslice : dataslice + self.seq_len]
                if self.recons_key in hf
                else None
            )

            if self.segmentation:
                if "lesion_segmentation" in hf:
                    seg_key = "lesion_segmentation"
                elif "segmentation" in hf:
                    seg_key = "segmentation"
                else:
                    seg_key = None

                segmentation = (
                    self.convert_mask(hf[seg_key][dataslice : dataslice + self.seq_len])
                    if seg_key
                    else torch.Tensor([])
                )
            else:
                segmentation = torch.Tensor([])

            attrs = dict(hf.attrs)
            attrs.update(metadata)

        if sensitivity_map.shape != kspace.shape:
            sensitivity_map = np.transpose(sensitivity_map, (2, 0, 1))

        if self.transform is not None:
            out = self.transform(
                kspace,
                sensitivity_map,
                mask,
                eta,
                target,
                attrs,
                fname.name,
                dataslice,
            )
        else:
            out = (
                kspace,
                sensitivity_map,
                mask,
                eta,
                target,
                attrs,
                fname.name,
                dataslice,
            )

        return (*out, segmentation)
