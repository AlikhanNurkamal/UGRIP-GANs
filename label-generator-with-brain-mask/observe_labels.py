# from custom_dataset import LabelsDataset
# import matplotlib.pyplot as plt
# import torch

# torch.set_printoptions(profile="full")

# dataset = LabelsDataset("/home/alikhan.nurkamal/brats-project/large-dataset/")

# label = dataset[0]
# print(label.shape)

# print(label[0, :, :, 7])

# # print(label[label == 1].size())

# fig, ax = plt.subplots(8, 8, figsize=(20, 20))
# for i in range(64):
#     ax[i // 8, i % 8].imshow(label[1, :, :, i], cmap="jet")
#     ax[i // 8, i % 8].axis("off")

# plt.show()

# from monai.transforms import \
#     Compose, AddChannel, ScaleIntensity, ToTensor, Resize, RandRotate, RandFlip, RandScaleIntensity, RandZoom, RandGaussianNoise, RandAffine, ResizeWithPadOrCrop
# from custom_dataset import LabelsDataset
# from torch.utils.data import DataLoader
# import torch
# import numpy as np


# def create_train_loader():
#     train_transforms = Compose([# RandRotate(prob=1, range_x=0.052, range_y=0.052, range_z=0.052),
#                                 # RandFlip(prob=1, spatial_axis=0),
#                                 # RandScaleIntensity(prob=0.1, factors=(0.1)),
#                                 # RandZoom(prob=1, min_zoom=(1.0), max_zoom=(1.1), mode="nearest"),
#                                 # RandGaussianNoise(prob=1, mean=0, std=0.01),
#                                 # ScaleIntensity(minv=0.0, maxv=1.0),
#                                 RandAffine(prob=1, translate_range=(4,4,0)),
#                                 ToTensor()])
    
#     train_ds = LabelsDataset(data_dir="/home/alikhan.nurkamal/brats-project/large-dataset/", transforms=train_transforms)
#     train_loader = DataLoader(train_ds, batch_size=16, shuffle=False, drop_last=True, num_workers=4, pin_memory=torch.cuda.is_available())
#     return train_loader

# train_loader = create_train_loader()

# for batch in train_loader:
#     print(batch.shape)
#     print(batch[0, 2, 30:40, 30:40, 32])
#     print(batch[0][torch.logical_and(batch[0] > 0, batch[0] < 1)])
#     break

# import torch
# import torch.nn.functional as F

# rand = torch.randn(16, 4, 64, 64, 64)
# out = F.softmax(rand, dim=1)
# res = torch.argmax(out, dim=1)

# print(res[0].shape)


# import nibabel
# import os
# import numpy as np
# import matplotlib.pyplot as plt

# def get_data(nifty, dtype="int16"):
#     if dtype == "int16":
#         data = np.abs(nifty.get_fdata().astype(np.int16))
#         data[data == -32768] = 0
#         return data
#     return nifty.get_fdata().astype(np.uint8)

# ROOT_DIR = "/home/alikhan.nurkamal/brats-project/large-dataset/"
# patient = "BraTS-GLI-00002-000"

# all_modality = nibabel.load(os.path.join(ROOT_DIR, patient, f"{patient}-all_modalities.nii.gz"))
# label_modality = nibabel.load(os.path.join(ROOT_DIR, patient, f"{patient}-seg.nii.gz"))

# all_image = get_data(all_modality, "int16")
# reshaped_image = np.max(all_image, axis=0)

# label = get_data(label_modality, "uint8")
# # Reshape the mask to have 5 channels
# reshaped_mask = np.zeros((5, 240, 240, 155))
# for i in range(4):
#     reshaped_mask[i] = (label == i).astype(int)
# reshaped_mask[4] = (reshaped_image > 0).astype(int)

# fig, ax = plt.subplots(10, 10, figsize=(20, 20))
# for i in range(100):
#     ax[i // 10, i % 10].imshow(reshaped_mask[4, :, :, i], cmap="gray")
#     ax[i // 10, i % 10].imshow(reshaped_mask[2, :, :, i], cmap="jet", alpha=0.3)
#     ax[i // 10, i % 10].axis("off")
# plt.show()


import torch
from WGAN_SigmaRat2 import Generator, Discriminator, Encoder, Code_Discriminator

# model_G = Generator(noise=500, channel=64)
# noise = torch.randn(2, 500, 1, 1, 1)
# output = model_G(noise)
# print(f"Generator output shape: {output.shape}")

# model_D = Discriminator()
# res = model_D(output).view(-1)
# print(f"Discriminator output shape: {res.shape}")

# model_E = Encoder(out_class=500)
# img = torch.randn(2, 5, 128, 128, 128)
# output = model_E(img).view(2, -1)
# print(f"Encoder output shape: {output.shape}")

# model_CD = Code_Discriminator(code_size=500, num_units=4096)
# res = model_CD(output)
# print(f"Code Discriminator output shape: {res.shape}")
# print(res.mean())
