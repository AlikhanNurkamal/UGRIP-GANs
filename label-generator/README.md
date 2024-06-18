# Tumor mask generation
I tried to generate brand new 3D tumor segmentation masks of sizes 64x64x64 using the WGAN_SigmaRat2 model by Ferreira et al. In his paper he used this model to generate synthetic rat brain MRI scans, but I slightly modified the networks definitions and final activation function of the Generator and Generator output post-processing steps. As a result, I was able to generate new synthetic brain tumor masks of shapes 64x64x64, which later could be used as input to a conditional GAN to generate synthetic brain MRI scans.

## Dataset
In order to train this model, I have used the BraTS-Glioma 2024 dataset, which consisted of 1251 brain MRI images with their corresponding tumor segmentation masks. However, these images have shape 240x240x155, that is why I cropped label images with the tumor at the center (check `custom_dataset.py`) to a shape of 64x64x64. Since some tumors are smaller than this shape, I added padding to obtain the desired shape.
