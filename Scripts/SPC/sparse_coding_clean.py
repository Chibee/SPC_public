import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.io as scipio
import os
import time
from sklearn import manifold
from tqdm.notebook import tqdm
from sparse_coding_helper_clean import *
np.random.seed(0)



def run_spc_experiments(codeword_dim, num_units, num_iter=1000, nt_max=1000, 
        batch_size=250, eps=1e-2, lr_r=1e-2, lr_Phi=1e-2, lmda=5e-3, lmda_test_list=[5e-3], 
        plot_spc_train=True, plot_style='relative', include_filler=False, mem_source_index=1, 
        mem_index=3, filler_as_target=False, flat_only=False, train_with_filler_only=True):
    start = time.time()
    # edit prefix to include learning rates of r and phi
    prefix = 'spc/' + str(codeword_dim) + '_' + str(num_units) + '_' + str(lmda).replace('.', '') + '_' + str(lr_r).replace('.', '') + '_' + str(lr_Phi).replace('.', '')
    compute_spc_distance(codeword_dim, num_units, num_iter, nt_max, 
        batch_size, eps, lr_r, lr_Phi, lmda, prefix, lmda_test_list, plot_spc_train, include_filler, 
        filler_as_target, flat_only, train_with_filler_only)

    end = time.time()
    print('run_spc_experiments took {:.2f} minutes'.format((end - start)/60))


