Data and code for: Isola, Xiao, Torralba, and Oliva. What makes an image memorable? CVPR 2011.

%%%%%%%%%% SETUP %%%%%%%%%%%
1. Download data from: http://web.mit.edu/phillipi/Public/WhatMakesAnImageMemorable/cvpr_memorability_data.zip
2. Download code from: http://web.mit.edu/phillipi/Public/WhatMakesAnImageMemorable/cvpr_memorability_code.zip
3. Place both downloads in same folder, which will be referred to as '.' below.
4. Download the LabelMe toolbox (http://labelme2.csail.mit.edu/Release3.0/browserTools/php/matlab_toolbox.php) and add this to ./Code/Library.


%%%%%%%%%% DATA %%%%%%%%%%%

%% Images: ./Data/Image data/target_images.mat and ./Data/Image data/filler_images.mat
Images used as targets and fillers.

%% Target image features: ./Data/Image data/target_features.mat
Features for each of our target images. Sorted in same order as target_images.mat. Please see './Code/Memorability code/Image features/demo.m' for documentation and demo visualizations.

Note that while we collected data on 2400 images, only the first 2222 were analyzed in the CVPR paper (the remaining images are textural and were excluded; the code provided will skip these).

Note also, we have not computed features for the filler images. Others are welcome to do so using ComputeImgFeats.m (see below).


%% Experiment results per image: ./Data/Experiment data/sorted_target_data.mat and ./Data/Experiment data/sorted_filler_data.mat
Structure arrays with entries for each of the target and filler images. Sorted in same order as target_images.mat and filler_images.mat.

Each entry has the following fields:
filename: matches name in SUN database
hits: number of times repeat of image was responded to
misses: number of times repeat of image was missed
false_alarms: number of times initial presentation of image was responded to
correct_rejections: number of times initial presentation of image was correctly not responded to

Note that for the fillers, hits and misses are calculated from the vigilance task (repeats only a few images after the initial presentation) whereas for the targets, hits and misses are calculated from the ~100-back target repeats.

%% Experiment results per subject: ./Data/Experiment data/subj_data.mat


%% Random splits of the data into training and testing sets: ./Data/Random splits
We used splits 1 through 25 for experiments in the CVPR paper (see code for details). (Note, however, that there was a bug in our paper's grid search splits for SVM hyperparameter selection -- named 'T1…' and 'T2…'; this is fixed in the splits included here. The rest of the splits are the same as used in the paper.)


%%%%%%%%%% CODE %%%%%%%%%%%

%% Running everything at once: ./run.m
This script will replicate all the results and figures in our paper, creating output in a new directory ./Analysis. By default, the script will use the same train/test splits we used in our experiments. (Note that there was a bug in our original SVM hyperparameter selection; this is fixed in the current code and causes the exact numbers resulting from running this code to differ a tiny bit from the published results.)

%% Computing features: ./Code/Memorability code/Image features/ComputeImgFeats.m
run.m uses the cached image features in ./Image data/img_data.mat, which is exactly what we used for our experiments. If you would instead like to modify the features, or compute features for a new set of images, please refer to ComputeImgFeats.m. Note that this script uses the LabelMe toolbox and so the results may vary slightly depending on the version of the toolbox you downloaded.

%% Predicting memorability: ./Code/Memorability code/Prediction/RunAllCVPRRegressions.m
RunAllCVPRRegressions.m tests a bunch of different conditions for predicting memorability from various features. If you would like to try new prediction schemes, you can try editing the parameters here (and in Prediction_5.m).

