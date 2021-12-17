import os

import numpy as np

import torch
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


class skmtea(Dataset):
    def __init__(self, split, data_root, mri_data_path, segmentation_path, data_files, seq_len=1, compact_masks=False, use_cache=True):
        self.split = split
        self.data_root = data_root
        self.mri_data_path = mri_data_path
        self.segmentation_path = segmentation_path
        self.seq_len = seq_len
        self.compact_masks = compact_masks
        
        cache_file = f'./.cache/skm_cache_{split}.pkl'
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
                mri_path = os.path.join(data_root, mri_data_path, file_name+".h5")
                mask_path = os.path.join(data_root, segmentation_path, file_name+".nii.gz")

                mri = h5py.File(mri_path, 'r')['target'].shape[2]
                mask = nib.load(mask_path).dataobj.shape[2]

                mri_num_slices = mri-self.seq_len+1
                mask_num_slices = mask-self.seq_len+1

                self.mri_slices += [(mri_path, i) for i in range(mri_num_slices)]
                self.mask_slices += [(mask_path, i) for i in range(mask_num_slices)]
            
            assert len(self.mri_slices) == len(self.mask_slices)
            
            dataset_cache.append(self.mri_slices)
            dataset_cache.append(self.mask_slices)
            
            if use_cache:
                os.makedirs('./.cache', exist_ok=True)
                with open(cache_file, "wb") as f:
                    pickle.dump(dataset_cache, f)
                print(f"Savinf cache to {cache_file}.")
                del dataset_cache

    def __len__(self):
        return len(self.mask_slices)

    def __getitem__(self, idx):
        fmri, mri_slice = self.mri_slices[idx]
        fmask, mask_slice = self.mask_slices[idx]
        
        mri = h5py.File(fmri, 'r', libver='latest')['target'][:, :, mri_slice:mri_slice+self.seq_len, :, :]
        mask = np.array(nib.load(fmask).dataobj[:, :, mask_slice:mask_slice+self.seq_len])

        mri_image = rss(to_tensor(mri), dim=-3)
        seg_mask = self.convert_mask(mask, compact=self.compact_masks)

        mri_image = rearrange(mri_image, 'x y z () i -> z i x y')
        seg_mask = rearrange(seg_mask, 'h w c s -> c s h w')
        
        return mri_image, seg_mask
    
    def convert_mask(self, mask, n_classes=7, compact=False):
        x, y, z = mask.shape
        out = np.zeros((x, y, z, n_classes), dtype=np.bool)
        for i in range(n_classes):
            out[:, :, :, i] = (mask == i)
        if compact:
            out[:, :, :, 3] = np.logical_or(out[:, :, :, 3], out[:, :, :, 4])
            out[:, :, :, 4] = np.logical_or(out[:, :, :, 5], out[:, :, :, 6])
            out = out[:, :, :, :5]
        return out.astype(np.uint8)


class DataModule(pl.LightningDataModule):
    def __init__(self, data_root, mri_data_path, segmentation_path, annotation_path=None, seq_len=1, use_cache=True, **kwargs):
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
                with open(os.path.join(data_root, annotation_path, split + ".json")) as f:
                    config = json.load(f)
                    self.file_split[split] = [i["file_name"][:-3] for i in config["images"]]

    def _make_dataset(self, split, data_files):
        return skmtea(split,
                      self.hparams.data_root,
                      self.hparams.mri_data_path,
                      self.hparams.segmentation_path,
                      data_files,
                      self.hparams.seq_len,
                      self.hparams.compact_masks,
                      self.hparams.use_cache)

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            if self.hparams.annotation_path:
                self.train = self._make_dataset("train", self.file_split)
                self.val = self._make_dataset("val", self.file_split)
            else:
                dataset = self._make_dataset("all", self.file_split)
                self.train, self.val = random_split(dataset, [int(self.hparams.train_fraction*len(dataset)),
                                                              len(dataset)-(int(self.hparams.train_fraction*len(dataset)))])
                self.val, self.test = random_split(self.val, [len(self.val)//2, len(self.val)-len(self.val)//2])

        if stage == "test" or stage is None:
            if self.hparams.annotation_path:
                self.test = self._make_dataset("test", self.file_split)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers)
    
    @staticmethod
    def add_data_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Dataset")

        # dataset arguments
        parser.add_argument("--data_root", default='data', type=str,
                            help="Path to the data root")
        parser.add_argument("--mri_data_path", default='raw_data', type=str,
                            help="Path to the raw mri data from the root.")
        parser.add_argument("--segmentation_path", default='segmentation_masks', type=str,
                            help="Path to the segmentation maps from the root.")
        parser.add_argument("--annotation_path", default=None, type=str,
                            help="Path to the annotation maps from the root. Optional")
        parser.add_argument("--seq_len", default=1, type=int,
                            help="Size of the slice looked at.")
        parser.add_argument("--train_fraction", default=0.8, type=float,
                            help="Fraction of th data used for training.") 
        parser.add_argument("--use_cache", default=True, type=bool,
                            help="Whether to cache dataset metadata in a pkl file")
        parser.add_argument("--compact_masks", default=False, type=bool,
                            help="Whether to cache dataset metadata in a pkl file")

        # data loader arguments
        parser.add_argument("--batch_size", default=1, type=int,
                            help="Data loader batch size")
        parser.add_argument("--num_workers", default=4, type=int,
                            help="Number of workers to use in data loader")

        return parent_parser