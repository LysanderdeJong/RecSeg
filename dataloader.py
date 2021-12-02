import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
import pickle
import nibabel as nib
from fastmri.data.transforms import to_tensor
from einops import rearrange

from utils import convert_mask


class skmtea(Dataset):
    def __init__(self, data_root, mri_data_path, segmentation_path, seq_len=1, use_cache=True):
        self.data_root = data_root
        self.mri_data_path = mri_data_path
        self.segmentation_path = segmentation_path
        self.seq_len = seq_len
        
        cache_file = './.cache/skm_cache.pkl'
        if os.path.isfile(cache_file) and use_cache:
            with open(cache_file, "rb") as f:
                dataset_cache = pickle.load(f)
            print("Using saved cache.")
        else:
            dataset_cache = []
            
        mri_files = os.listdir(os.path.join(data_root, mri_data_path))
        mri_files.sort()
        segmentation_masks = os.listdir(os.path.join(data_root, segmentation_path))
        segmentation_masks.sort()
        
        assert len(mri_files) == len(segmentation_masks)
        
        if os.path.isfile(cache_file) and use_cache:
            self.mri_slices = dataset_cache[0]
            self.mask_slices = dataset_cache[1]
        else:
            self.mri_slices = []
            self.mask_slices = []

            print("Generating cache file.")

            for mri_file, mask_file in zip(mri_files, segmentation_masks):
                mri_path = os.path.join(data_root, mri_data_path, mri_file)
                mask_path = os.path.join(data_root, segmentation_path, mask_file)

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
        
        mri = h5py.File(fmri, 'r', libver='latest')['target'][:, :, mri_slice:mri_slice+self.seq_len, 0, :]
        mask = np.array(nib.load(fmask).dataobj[:, :, mask_slice:mask_slice+self.seq_len])

        mri_image = to_tensor(mri)
        seg_mask = convert_mask(mask)

        mri_image = rearrange(mri_image, 'x y z () i -> z i x y')
        seg_mask = rearrange(seg_mask, 'h w c s -> c s h w')
        
        return mri_image, seg_mask


class DataModule(pl.LightningDataModule):
    def __init__(self, data_root, mri_data_path, segmentation_path, seq_len=1, use_cache=True, **kwargs):
        super().__init__()
        self.save_hyperparameters()

    def _make_dataset(self):
        return skmtea(self.hparams.data_root,
                      self.hparams.mri_data_path,
                      self.hparams.segmentation_path,
                      self.hparams.seq_len,
                      self.hparams.use_cache)

    def setup(self, stage=None):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            dataset = self._make_dataset()
            self.train, self.val = random_split(dataset, [int(self.hparams.train_fraction*len(dataset)),
                                                          len(dataset)-(int(self.hparams.train_fraction*len(dataset)))])

        # Assign test dataset for use in dataloader(s)
        # if stage == "test" or stage is None:
        #     self.test = None
        #     self.dims = tuple(self.mnist_test[0][0].shape)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.hparams.batch_size,
                          shuffle=True, num_workers=self.hparams.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.hparams.batch_size,
                          shuffle=False, num_workers=self.hparams.num_workers)

    # def test_dataloader(self):
    #     return DataLoader(self.test, batch_size=self.hparams.batch_size,
    #                       shuffle=False, num_workers=self.hparams.num_workers)
    
    @staticmethod
    def add_data_specific_args(parent_parser):  # pragma: no-cover
        parser = parent_parser.add_argument_group("Dataset")

        # dataset arguments
        parser.add_argument("--data_root", default='data', type=str,
                            help="Path to the data root")
        parser.add_argument("--mri_data_path", default='raw_data', type=str,
                            help="Path to the raw mri data from the root.")
        parser.add_argument("--segmentation_path", default='segmentation_masks', type=str,
                            help="Path to the segmentation maps from the root.")
        parser.add_argument("--seq_len", default=1, type=int,
                            help="Size of the slice looked at.")
        parser.add_argument("--train_fraction", default=0.8, type=float,
                            help="Fraction of th data used for training.") 
        parser.add_argument("--use_cache", default=True, type=bool,
                            help="Whether to cache dataset metadata in a pkl file")

        # data loader arguments
        parser.add_argument("--batch_size", default=1, type=int,
                            help="Data loader batch size")
        parser.add_argument("--num_workers", default=4, type=int,
                            help="Number of workers to use in data loader")

        return parent_parser