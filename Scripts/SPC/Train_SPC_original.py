import os, sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import scipy.io as scipio
np.random.seed(0)
import importlib
import sparse_coding_helper_clean as spch
import warnings
warnings.filterwarnings(action='once')

target_img_list = list(range(1, 2223))
folder_list = ['Targets', 'Fillers']

layer_num = int(sys.argv[1])
keras_layer_list = ['block1_pool', 'block2_pool', 'block3_pool', 'block4_pool', 'block5_pool', 'fc1', 'fc2']
layer = keras_layer_list[layer_num-1]
print(layer)
train_with_filler_only = False

dir_list = [os.path.join('../../Activations', layer, x) for x in folder_list]
codeword_dim = 1000
num_codewords = 500
distance_rec_quality_l = spch.compute_pairwise_spc_distance(layer, dir_list,
        target_img_list, codeword_dim, num_codewords,
        include_filler=True,
        train_with_filler_only=train_with_filler_only)

save_dir = os.path.join('../../Results', 'RE',
                    layer)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir, exist_ok=True)
np.save(os.path.join(save_dir, 'spc_rec_quality'), distance_rec_quality_l)


