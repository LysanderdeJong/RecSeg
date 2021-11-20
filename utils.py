import nibabel as nib

def read_nii(file_path):
    nii = nib.load(file_path)
    return nii.get_fdata()