clear all; close all;

source_folder = '../../../data/VOC_arg/SegmentationClass';
target_folder = '../../../data/VOC_arg/SegmentationClass_label';

imgs_dir = dir(fullfile(source_folder, '*.png'));

if ~exist(target_folder, 'dir')
    mkdir(target_folder)
end

for i = 1 : length(imgs_dir)
    fprintf('processing %d/%d\n', i, length(imgs_dir));
    img = imread(fullfile(source_folder, imgs_dir(i).name));
    imwrite(img, fullfile(target_folder, imgs_dir(i).name));
end
