import os

# import bart
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt

from mridc.collections.reconstruction.data.subsample import Poisson2DMaskFunc
from mridc.collections.reconstruction.parts.utils import apply_mask


def pics_recon(kspace, sensitivity_maps, reg_wt=0.005, num_iters=60):
    return bart.bart(
        1,
        f"pics -d0 -S -R W:7:0:{reg_wt} -i {num_iters}",
        kspace,
        sensitivity_maps,
    )[0]


mask_gen = Poisson2DMaskFunc([0.7], [8])

test_file = h5py.File(
    "/data/projects/recon/data/public/qdess/v1-release/files_recon_calib-24/MTR_223.h5"
)
print(test_file.keys())

slice_indx = 300

from mridc.collections.common.parts.fft import ifft2c, fft2c, ifftshift, fftshift

kspace = test_file["kspace"][:, :, :, 0, :]
kspace = torch.from_numpy(kspace)
# kspace = kspace[slice_indx]
# kspace = np.expand_dims(kspace, axis=0)
kspace = ifftshift(kspace, dim=[1, 2])
imspace = torch.fft.ifftn(kspace, dim=[1, 2])
# kspace = fftshift(kspace, dim=[1, 2])
imspace = torch.permute(imspace, dims=(0, 3, 1, 2))
kspace = torch.permute(kspace, dims=(0, 3, 1, 2)).numpy()
# kspace, mask, acc = apply_mask(
#     torch.view_as_real(torch.from_numpy(kspace)), mask_func=mask_gen
# )
# mask = mask.squeeze().numpy()
# kspace = torch.view_as_complex(kspace).numpy()
# kspace = np.transpose(kspace, axes=(0, 2, 3, 1))

imspace = imspace / torch.amax(torch.abs(imspace)).numpy()
# kspace = np.fft.fftn(imspace, axes=(-2, -1))
plt.subplot(2, 4, 1)
plt.imshow(np.abs(imspace[slice_indx, 0]), cmap="gray")
plt.subplot(2, 4, 2)
plt.imshow(np.angle(imspace[slice_indx, 0]), cmap="gray")
plt.subplot(2, 4, 3)
plt.imshow(np.real(imspace[slice_indx, 0]), cmap="gray")
plt.subplot(2, 4, 4)
plt.imshow(np.imag(imspace[slice_indx, 0]), cmap="gray")
plt.subplot(2, 4, 5)
plt.imshow(np.abs(imspace[:, 0, :, 70]), cmap="gray")
plt.subplot(2, 4, 6)
plt.imshow(np.angle(imspace[:, 0, :, 70]), cmap="gray")
plt.subplot(2, 4, 7)
plt.imshow(np.real(imspace[:, 0, :, 70]), cmap="gray")
plt.subplot(2, 4, 8)
plt.imshow(np.imag(imspace[:, 0, :, 70]), cmap="gray")
plt.show()

sense_maps = test_file["maps"][slice_indx, :, :, :, 0]
sense_maps = np.expand_dims(sense_maps, axis=0)
sense_maps = np.transpose(sense_maps, axes=(0, 3, 1, 2))
sense_maps = sense_maps / np.sum(np.abs(sense_maps), 1, keepdims=True)
sense_maps = sense_maps / np.max(np.abs(sense_maps))

# sense_maps = np.fft.fftn(sense_maps, axes=(-2, -1))
# sense_maps = np.fft.fftshift(sense_maps, axes=(-2, -1))
# sense_maps = np.fft.ifftn(sense_maps, axes=(-2, -1))

# sense_maps = np.fft.fftshift(sense_maps, axes=(-2, -1))
# sense_maps = np.transpose(sense_maps, axes=(0, 3, 1, 2))

# recon = pics_recon(kspace, sense_maps)
# recon = np.fft.fftshift(recon, axes=(-2, -1))

# sense_maps_norm = sense_maps / np.sum(np.abs(sense_maps), axis=(-1), keepdims=True)
# recon2 = pics_recon(kspace, sense_maps_norm, reg_wt=0.0005)
# recon2 = np.fft.fftshift(recon2, axes=(-2, -1))

# sense_maps_norm = sense_maps / np.amax(np.abs(sense_maps))
# recon3 = pics_recon(kspace, sense_maps_norm)
# recon3 = np.fft.fftshift(recon3, axes=(-2, -1))

# print(kspace.shape, sense_maps.shape, recon.shape, recon2.shape)
mask = np.abs(kspace)[slice_indx, 0, :, :]
recon = np.fft.ifftn(kspace[slice_indx : slice_indx + 1], axes=(-2, -1))
recon2 = np.fft.ifftshift(recon, axes=(-2, -1))
recon3 = np.sum(recon * sense_maps.conj(), 1)[0, :, :]

plt.subplot(1, 6, 1)
ax = plt.imshow(mask, cmap="gray")
# plt.colorbar(ax)
plt.title("mask")
plt.subplot(1, 6, 2)
ax = plt.imshow(np.abs(sense_maps)[0, 0, :, :], cmap="gray")
# plt.colorbar(ax)
plt.title("sensitivity maps")
plt.subplot(1, 6, 3)
ax = plt.imshow(np.abs(recon)[0, 0, :, :], cmap="gray")
# plt.colorbar(ax)
plt.title("CS")
plt.subplot(1, 6, 4)
ax = plt.imshow(np.abs(recon2)[0, 0, :, :], cmap="gray")
# plt.colorbar(ax)
plt.title("CS norm sum")
plt.subplot(1, 6, 5)
ax = plt.imshow(np.abs(recon3), cmap="gray")
# plt.colorbar(ax)
plt.title("CS norm max")
plt.subplot(1, 6, 6)
ax = plt.imshow(np.abs(test_file["target"][slice_indx, :, :, 0, 0]), cmap="gray")
# plt.colorbar(ax)
plt.title("target")
plt.suptitle(f"DMF007_T2_AXFLAIR_transverse.h5_{slice_indx}")
plt.savefig("/home/lgdejong/Pictures/pics_test.png")
plt.show()
