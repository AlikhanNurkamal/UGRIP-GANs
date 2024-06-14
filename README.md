# UGRIP-GANs
This repository shows all my work on Generative Adversarial Networks that has been done during a one-month internship at MBZUAI in Abu Dhabi:
- `label-generator/` uses the code from <a href="https://doi.org/10.3390/app12104844">Generation of Synthetic Rat Brain MRI Scans with a 3D Enhanced Alpha Generative Adversarial Network</a> by Ferreira et al. with slight modifications in models to generate brain tumor segmentation masks.
- `label-generator-with-brain-mask/` also uses the code from above mentioned paper but with additional layers in model definitions in order to generate brain tumor segmentation masks taking into account the mask of a brain.
- `patch-label-generator` uses the code from <a href="">Generating 3D TOF-MRA volumes and segmentation labels using generative adversarial networks</a> to generate both brain MRI patches with their corresponding tumor segmentation masks by applying a similar idea that is presented in the paper.

