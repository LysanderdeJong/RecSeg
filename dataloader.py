import os
import numpy as np
import torch
import h5py
from torch.utils.data import Dataset
from einops import rearrange

from utils import read_nii, convert_mask


class skmtea(Dataset):
    def __init__(self, data_root, mri_data_path, segmentation_path, seq_len=1):
        mri_files = os.listdir(os.path.join(data_root, mri_data_path))
        segmentation_masks = os.listdir(os.path.join(data_root, segmentation_path))
        
        assert len(mri_files) == len(segmentation_masks)
        
        self.mri_slices = []
        self.mask_slices = []
        
        for mri_file, mask_file in zip(mri_files, segmentation_masks):
            mri_path = os.path.join(data_root, mri_data_path, mri_file)
            mask_path = os.path.join(data_root, segmentation_path, mask_file)
            
            mri = h5py.File(mri_path)['target'][:, :, :, 0, :]
            mask = convert_mask(read_nii(mask_path))
            
            slices = mri.shape[2]
            
            for i in range(slices-seq_len+1):
                self.mri_slices.append(np.abs(mri[:, :, i:i+seq_len, :]))
                self.mask_slices.append(mask[:, :, i:i+seq_len, :])
            
        assert len(self.mri_slices) == len(self.mask_slices)

    def __len__(self):
        return len(self.mask_slices)

    def __getitem__(self, idx):
        mri_image = self.mri_slices[idx]
        seg_mask = self.mask_slices[idx]
        
        mri_image = rearrange(mri_image, 'x y z e -> (z e) x y')
        seg_mask = rearrange(seg_mask, 'h w c s -> (c s) h w')
        
        return mri_image, seg_mask