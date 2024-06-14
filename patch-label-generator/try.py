import config as c
import torch
from model import Generator, Discriminator
import nibabel
# from pytorch_model_summary import summary
import numpy as np
import matplotlib.pyplot as plt
import gzip
import utils as ut

# model_G = Generator()
# noise = torch.randn(c.batch_size, c.nz, 1, 1, 1)
# output = model_G(noise).detach().cpu()

# print(f"Generator output shape: {output.shape}")

# model_D = Discriminator()
# res = model_D(output).view(-1)
# print(f"Discriminator output shape: {res.shape}")

# print(summary(model_G, torch.zeros(c.batch_size, c.nz, 1, 1, 1)))

# print(summary(model_D, torch.zeros(c.batch_size, c.nc, c.image_size[0],
#                                 c.image_size[1], c.image_size[2])))

# from dataset import GANDataset
# import glob

# path_patches = glob.glob(c.dataroot+"train/patches/*_random_*.gz")
# path_labels = glob.glob(c.dataroot+"train/seg_labels/*_random_*.gz")
# ds = GANDataset(path_patches, path_labels)

# print(len(ds))

# import numpy as np

# data = np.load('./checkpoints/results/trial_1/Wasserstein_D.npy')
# print(data)


# model_G = Generator()
# saved_params = torch.load('./checkpoints/models/trial_1/epoch_4.pth', map_location='cpu')
# model_G.load_state_dict(saved_params['Generator_state_dict'])

# noise = torch.randn(c.batch_size, c.nz, 1, 1, 1)
# output = model_G(noise).detach()

# sample_idx = [0, 1]  # because batch size is 2

# for idx in sample_idx:
#     # hard thresholding for visualisation
#     sample = output[idx].clone()
#     print(f"Patch shape: {sample[:4].shape}")
#     print(f"Label shape: {sample[4:].shape}")


# def get_data(nifty, dtype="int16"):
#     if dtype == "int16":
#         data = np.abs(nifty.get_fdata().astype(np.int16))
#         data[data == -32768] = 0
#         return data
#     return nifty.get_fdata().astype(np.uint8)

ROOT_DIR = "/home/alikhan.nurkamal/brats-project/gans-for-brats/3DGAN_synthesis_of_3D_TOF_MRA_with_segmentation_labels/checkpoints/results/trial_2/"
# ROOT_DIR = "/home/alikhan.nurkamal/Downloads/"
modalities = nibabel.load(ROOT_DIR + "fake_while_training_epoch_100_sample_0_patch.nii.gz")
patch = modalities.get_fdata()
affine, header = modalities.affine, modalities.header

labels = nibabel.load(ROOT_DIR + "fake_while_training_epoch_100_sample_0_label.nii.gz")
label = labels.get_fdata()
affine_label, header_label = labels.affine, labels.header

label[label > c.gen_threshold] = 1
label[label <= c.gen_threshold] = 0
patch = ut.rescale_unet(patch)  # rescaling back to 0-255

t2f = patch[0]  # FLAIR modality
whole_tumor = label[1]  # whole tumor label

# fig, ax = plt.subplots(8, 8, figsize=(20, 20))
# for i in range(64):
#     ax[i//8, i%8].imshow(t2f[:, :, i], cmap="gray")
#     ax[i//8, i%8].axis("off")

# plt.show()

img = nibabel.nifti1.Nifti1Image(t2f, affine=affine)
nibabel.save(img, "./patch_t2f.nii.gz")

label_img = nibabel.nifti1.Nifti1Image(whole_tumor, affine=affine_label)
nibabel.save(label_img, "./label_wt.nii.gz")


# ROOT_DIR = "/home/alikhan.nurkamal/brats-project/large-dataset-patches/10_ppp_128x128x128/train/"
# patch_file = gzip.GzipFile(ROOT_DIR + "seg_labels/BraTS-GLI-00009-000_label_random_3.npy.gz", "r")
# patch = np.load(patch_file)
# print(patch.shape)
# img = nibabel.nifti1.Nifti1Image(patch[0], affine=np.eye(4))
# nibabel.save(img, "./patch_real.nii.gz")


# gen_losses = np.load("./checkpoints/results/trial_1/D_losses.npy")
# print(gen_losses)
