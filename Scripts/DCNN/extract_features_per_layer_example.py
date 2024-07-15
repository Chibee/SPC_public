import pandas as pd
import numpy as np
from PIL import Image
import sys
sys.path.append('../../Keras-VGG16-places365') #### Change this to where you put the Places model
import os, glob
from cv2 import resize
from vgg16_hybrid_places_1365 import VGG16_Hybrid_1365
from keras.models import Model

# read inputs
layer_ind = int(sys.argv[1])-1
img_top_dir = '../../Images'
layer_list = ['block1_pool','block2_pool','block3_pool','block4_pool','block5_pool','fc1','fc2']
layer_name = layer_list[layer_ind] # the layers I used:
img_ind = '1032'
output_dir = '../../Activations'
if not os.path.isdir(os.path.join(output_dir, layer_name)):
    os.mkdir(os.path.join(output_dir, layer_name))

def extract_activation(img_dir,base_model,layer_name):
    # Specify the layer
    model = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)

    # Load and resize the image
    curr_img = Image.open(os.path.join(img_dir))
    curr_img= np.array(curr_img, dtype=np.uint8)
    curr_img = resize(curr_img, (224, 224))
    curr_img = np.expand_dims(curr_img, 0)
    
    curr_img_activations = model.predict(curr_img)

    return curr_img_activations

# Load model
base_model = VGG16_Hybrid_1365(weights='places') # You can use base_model.summary() to see the information about all the layers
# Go through the Target and Filler images
type_list = ['Targets', 'Fillers']
for type in type_list:
    type_dir = os.path.join(img_top_dir, type)
    img_list = glob.glob(os.path.join(type_dir, '*.jpg'))
    for curr_img in img_list:
        img_ind = os.path.basename(curr_img).split('.')[0]
        if (type == 'Targets') & (int(img_ind) >= 2223):
            continue
        else:
            activation_array = extract_activation(curr_img, base_model, layer_name)

            # Save the activation
            if not os.path.isdir(os.path.join(output_dir, layer_name, type)):
                os.mkdir(os.path.join(output_dir, layer_name, type))
            np.save(os.path.join(output_dir, layer_name, type, '{}.npy'.format(img_ind)), activation_array)




