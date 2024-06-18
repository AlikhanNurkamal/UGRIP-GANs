import config as c
import nibabel
import numpy as np
import gzip
import utils as ut


GENERATED_OUT_DIR = "./"
modalities = nibabel.load(GENERATED_OUT_DIR + "fake_while_training_epoch_100_sample_0_patch.nii.gz")
patch = modalities.get_fdata()

labels = nibabel.load(GENERATED_OUT_DIR + "fake_while_training_epoch_100_sample_0_label.nii.gz")
label = labels.get_fdata()

label[label > c.gen_threshold] = 1  # thresholding
label[label <= c.gen_threshold] = 0
patch = ut.rescale_unet(patch)  # rescaling back to 0-255

t2f = patch[0]  # FLAIR modality
t1n = patch[1]  # T1 modality
t1c = patch[2]  # T1c modality
t2w = patch[3]  # T2 modality

tumor_core = label[0]  # tumor core label
whole_tumor = label[1]  # whole tumor label
enhan_tumor = label[2]  # enhancing tumor label

img = nibabel.nifti1.Nifti1Image(t2f, affine=modalities.affine)
nibabel.save(img, "./patch_t2f.nii.gz")

label_img = nibabel.nifti1.Nifti1Image(whole_tumor, affine=labels.affine)
nibabel.save(label_img, "./label_wt.nii.gz")

# The code below is to observe the original patch or label file and
# save it as a nifti file for visualization

# PATCHES_DIR = "./"
# patch_file = gzip.GzipFile(PATCHES_DIR + "seg_labels/BraTS-GLI-00009-000_label_random_3.npy.gz", "r")
# patch = np.load(patch_file)
# img = nibabel.nifti1.Nifti1Image(patch[0], affine=np.eye(4))
# nibabel.save(img, "./patch_real.nii.gz")
