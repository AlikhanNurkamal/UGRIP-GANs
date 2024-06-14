# UGRIP-GANs
I would like to express my sincere gratitude to my mentors Sarim Hashmi, Sanoojan Baliah, and Fadillah Adamsyah Maani, who were helping and supporting us greatly throughout the whole internship program.

## Motivation
Under supervision of our PI Dr. Mohammad Yaqub, our team is taking part in BraTS 2024 competition (BraTS stands for Brain Tumor Segmentation). However, since the datasets in this competition are quite small, it is unreasonable to use large models (i.e. ViTs or Mambas) as they are prone to overfitting when training on small datasets. In order to overcome this problem it was decided to generate synthetic data (both brain MRI images and their tumor segmentation masks) using GANs and Diffusion Models. I was working on 3D GANs and my main goal was to generate 3D brain MRI images and their ground truth tumor masks.

## Ideas
While discovering 3D GANs, the following ideas have been considered:
1. Use true 3D tumor masks from the dataset and train a GAN to generate similar tumor masks. Since the original segmentation masks had a shape of 240x240x155, it was decided to crop only the tumorous part of a mask of shape 64x64x64. If a tumor was smaller than 64x64x64, it was padded to the desired shape.
2. Use not only true 3D tumor masks from the dataset (as stated above), but also use the segmentation mask of a brain. ADD MORE LATER.
3. Generate both a patch (of shape 128x128x128) of a brain MRI image - that includes all 4 modalities of an MRI scan - and its tumor mask - that includes 3 labels: Enhancing tumor, Tumor core, and Whole tumor. ADD MORE LATER.

## My work
This repository shows all my work on Generative Adversarial Networks that has been done during a one-month internship at MBZUAI in Abu Dhabi:
- `label-generator/` uses the code from <a href="https://doi.org/10.3390/app12104844">Generation of Synthetic Rat Brain MRI Scans with a 3D Enhanced Alpha Generative Adversarial Network</a> by Ferreira et al. with slight modifications in models to generate brain tumor segmentation masks.
- `label-generator-with-brain-mask/` also uses the code from above mentioned paper but with additional layers in model definitions in order to generate brain tumor segmentation masks taking into account the mask of a brain.
- `patch-label-generator` uses the code from <a href="https://doi.org/10.1016/j.media.2022.102396">Generating 3D TOF-MRA volumes and segmentation labels using generative adversarial networks</a> to generate both brain MRI patches with their corresponding tumor segmentation masks by applying a similar idea that is presented in the paper.
