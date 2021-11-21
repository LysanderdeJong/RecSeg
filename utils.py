import nibabel as nib
import numpy as np

def read_nii(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata()

def get_model():
    pass

def convert_mask(mask, n_classes=7):
    x, y, z = mask.shape
    out = np.zeros((x, y, z, n_classes), dtype=np.bool)
    for i in range(n_classes):
        out[:, :, :, i] = (mask == i)
    return out.astype(np.int8)