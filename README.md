# SPC_public

This repo contains the basic codes for extracting distinctiveness and training the sparse coding models in the following paper. Lin, Q., Li, Z., Lafferty, J., & Yildirim, I. (2024). Images with harder-to-reconstruct visual representations leave stronger memory traces. Nature Human Behaviour, 1-12. The initial version of the sparse coding part was written by Zifan Li and later modified by Qi Lin.

## Setup and downloading
1. Clone this repo
2. Install the conda environment
3. Download the Isola dataset from http://web.mit.edu/phillipi/Public/WhatMakesAnImageMemorable/ and place it under ./IsolaEtAl/
4. Download the VGG-16 model used in the paper into the top directory of this repo (i.e., at the same level of Scripts/Images) from: https://github.com/GKalliatakis/Keras-VGG16-places365. I am using the Hybrid 1365 classes model which is trained on both ImageNet images and their Places dataset and had the highest overall classification results.

## Preparation
1. Convert the images: The image data in Isola et al. (2014) were stored in matlab files so we should convert them into jpg files for ease of viewing and later processing. You can use ./Scripts/Prep/Save_images_from_Isola.m or write your own python version to do so. The resulting jpg files should be stored in:
./Images/Targets/: 

this folder includes images used as targets in the original Isola et al. study. Only the first 2222 images are the actual targets (i.e., those that came with memorability scores).  

./Images/Fillers/: 

this folder includes images used as fillers in the Isola study (e.g., used as attention check or fillers that never repeated). 

2. Extract image features from various layers of VGG
For extracting features from layers 1-7 (corresponding to the 5 maxpooling convolutional layers and 2 fully connected layers) referred to in the paper, run the following function in ./Script/DCNN (using layer 7 as an example):
```
python -r extract_features_per_layer_example.py 7
```
When I tried to run it on our cluster, tensorflow sometimes complained about not being able to find the relevant GPU but for the purpose of extracting activations, it was usually fine (i.e., you can ignore those errors). 
The resulting activations should be stored in ./Activations/#layername

## Calculating distinctiveness


