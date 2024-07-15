import pandas as pd
import numpy as np
import sys, os, glob, math, random, time

# read the input
curr_ioi = int(sys.argv[1])
layer_ind = int(sys.argv[2])-1
layer_list = ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool','fc1','fc2']
curr_layer = layer_list[layer_ind]


# specify the directories
output_dir = '../../Results/Dist'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
activation_dir = '../../Activations'

curr_img_df = pd.DataFrame(columns=['img_of_interest', 'img_to_compare', 'aggregate_dist'])

curr_ioi_feature = np.squeeze(np.load(os.path.join(activation_dir, curr_layer, 'Targets',
                                                   '{}.npy'.format(curr_ioi))))
# first compare target image set
img_list = glob.glob(os.path.join(activation_dir, curr_layer, 'Targets', '*.npy'))
for comp_img in img_list:
    curr_comp_img_feature = np.squeeze(np.load(comp_img))

    dist_aggregate = math.sqrt(
        np.sum((curr_comp_img_feature - curr_ioi_feature) ** 2))  #calculate the euclidean distance

    curr_comp_img_ind = comp_img.split('/')[-1].split('.')[0]

    curr_img_df.loc[len(curr_img_df)] = {'img_of_interest': curr_ioi,
                                         'img_to_compare': curr_comp_img_ind,
                                         'dist': dist_aggregate}

# now compare the filler image set
img_list = glob.glob(os.path.join(activation_dir, curr_layer, 'Fillers', '*.npy'))
for comp_img in img_list:
    curr_comp_img_feature = np.squeeze(np.load(comp_img))

    dist_aggregate = math.sqrt(np.sum((curr_comp_img_feature - curr_ioi_feature) ** 2))

    curr_comp_img_ind = comp_img.split('/')[-1].split('.')[0]

    curr_img_df.loc[len(curr_img_df)] = {'img_of_interest': curr_ioi,
                                         'img_to_compare': curr_comp_img_ind + '_filler',
                                         'aggregate_dist': dist_aggregate}

# Now find the distance to the nearest neighbor
# Need to get rid of dist = 0 (i.e., those are identical images)
filtered_img_df = curr_img_df[curr_img_df['aggregate_dist'] > 0].sort_values(by=['aggregate_dist']).reset_index(
    drop=True)
NN_img = filtered_img_df.iloc[0]

# Write the results
output_file = os.path.join(output_dir, '{}_NN.csv'.format(curr_layer))
#This is to prevent too many accesses to the same csv file when running jobs in parallel
time.sleep(100 * random.random())
with open(output_file, 'a') as f:
    f.write('{}, {}, {}\n'.format(curr_ioi, NN_img['img_to_compare'], NN_img['aggregate_dist']))
