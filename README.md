# SPC_public

This repo contains the basic codes for extracting distinctiveness and training the sparse coding models in the following paper. Lin, Q., Li, Z., Lafferty, J., & Yildirim, I. (2024). Images with harder-to-reconstruct visual representations leave stronger memory traces. Nature Human Behaviour, 1-12. The initial version of the sparse coding part was written by Zifan Li and later modified by Qi Lin.

## Setup and downloading
1. Clone this repo
2. Install the conda environment
3. Download the Isola dataset from http://web.mit.edu/phillipi/Public/WhatMakesAnImageMemorable/ and place it under ./IsolaEtAl/
4. Download the VGG-16 model used in the paper into the top directory of this repo (i.e., at the same level of Scripts/Images) from: https://github.com/GKalliatakis/Keras-VGG16-places365. I am using the Hybrid 1365 classes model which is trained on both ImageNet images and their Places dataset and had the highest overall classification results.

## Preparation
The image data in Isola et al. (2014) were stored in matlab files so we should convert them into jpg files for ease of viewing and later processing. You can use ./Scripts/Prep/

