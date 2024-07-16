% Load and convert the target images 
load('../../IsolaEtAl/Image data/target_images.mat')
img_dir = '../../Images_2/Targets/';

% Note that there are 2400 images in the img matrix but only the first 2222
% are of interest. The remaining are texturized images. 
for curr_img_ind = 1:2222
    curr_img = img(:,:,:,curr_img_ind);
    imwrite(curr_img, [img_dir num2str(curr_img_ind) '.jpg'],'jpg');
end

clear img
% Load and convert the filler images 
load('../../IsolaEtAl/Image data/filler_images.mat')
filler_img_dir = '../../Images_2/Fillers/';

num_fillers = size(img,4);

for curr_img_ind = 1:num_fillers
    curr_img = img(:,:,:,curr_img_ind);
    imwrite(curr_img, [filler_img_dir num2str(curr_img_ind) '.jpg'],'jpg');
end