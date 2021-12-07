import numpy as np
import torch
from einops import rearrange


def convert_mask(mask, n_classes=7, compact=False):
    x, y, z = mask.shape
    out = np.zeros((x, y, z, n_classes), dtype=np.bool)
    for i in range(n_classes):
        out[:, :, :, i] = (mask == i)
    if compact:
        out[:, :, :, 3] = np.logical_or(out[:, :, :, 3], out[:, :, :, 4])
        out[:, :, :, 4] = np.logical_or(out[:, :, :, 5], out[:, :, :, 6])
        out = out[:, :, :, :5]
    return out.astype(np.int8)

def segmentation_volume_to_img(seg):
    if len(seg.shape) == 4:
        c_dim = 1
    elif len(seg.shape) == 3:
        c_dim = 0
    else:
        raise ValueError

    if isinstance(seg, np.ndarray):
        img = np.argmax(seg, axis=c_dim)
        img = rearrange(img, 'c h w -> h w c')
    elif isinstance(seg, torch.Tensor):
        img = torch.argmax(seg, dim=c_dim)
    return img

def get_model(**kwargs):
    pass

def get_datase(**kwargs):
    pass