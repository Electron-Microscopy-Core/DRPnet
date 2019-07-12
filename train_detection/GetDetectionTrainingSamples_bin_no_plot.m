%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% Program Name: DRPnet Particle Picking
%
%  Filename: GetTrainingDetectionSamples_bin_no_plot.m
%
%  Description: 
%        Input:
%       Output:
%
%  Author: Nguyen Phuoc Nguyen
%
%  Copyright (C) 2018-2019. 
%       Nguyen Phuoc Nguyen, Ilker Ersoy, Filiz Bunyak, 
%       Tommi A. White, and Curators of the
%       University of Missouri, a public corporation.
%       All Rights Reserved.
%
%  Created by:
%     Nguyen Phuoc Nguyen, Ilker Ersoy, Filiz Bunyak, Tommi A. White
%     Dept. of Biochemistry & Electron Microscopy Core
%     and Dept. of Electrical Engineering and Computer Science,
%     University of Missouri-Columbia.
%
%  For more information, contact:
%     Dr. Tommi A. White
%     W117 Veterinary Medicine Building
%     University of Missouri, Columbia
%     Columbia, MO 65211
%     (573) 882-8304
%     whiteto@missouri.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all;
close all;
set(0,'DefaultFigureWindowStyle','docked');
% warning('off','all');
clc;

addpath('..\source');
addpath('..\utilities_fb');



%% SET UP PARAMETERS

% input_file = '../input/10061_params_TEST.txt';
% input_file = '../input/10017_params_TEST.txt';
% input_file = '../input/10005_params_TEST.txt';

% input_file = '../input/OKT3_params_TRAIN_TEST.txt';
% input_file = '../input/10061_params_TRAIN_TEST.txt';
% input_file = '../input/10017_params_TRAIN_TEST.txt';
input_file = '../input/10005_params_TRAIN_TEST.txt';

[names vals inputParams]=ParseParams(input_file);

for i=1:length(names)
    if isnumeric(vals{i})
        eval([genvarname(names{i}) ' = ' num2str(vals{i}) ';']);
    else
        eval([genvarname(names{i}) ' = ' vals{i} ';']);
    end
end

[scale_factor, rbox_scale, sigma_gauss, f3] = SetScaleFactors(inpath, box_size);

groundtruth_path = 'W:\DATA2\CRYO\TRPV1-EMPIAR10005\TRPV1_coords_35645';
% groundtruth_path = 'W:\DATA2\CRYO\TRPV1-EMPIAR10005\TRPV1_coords_88915';
% groundtruth_path = 'W:\DATA2\CRYO\TRPV1-EMPIAR10005\trpv_full_relion_GT\Micrographs';

start_mic_train_detect = 101;
num_train_mics_detect = 49;

%%

% rbox_scale = 30;

% rad=10;
% rad=16;
% rad=20;


% radius = 20;


r_patch = rbox_scale;

%%



patches = [];
images.data = [];
images.labels = [];




file_list = dir(fullfile(inpath, '*.mrc'));
total_mics = numel(file_list);

for i = start_mic_train_detect:(start_mic_train_detect + num_train_mics_detect - 1)

    fname = file_list(i).name;
%     im = PreprocessImage_2(fullfile(inpath,fname), scale_factor, is_negative_stain_data);
    im = PreprocessImage(fullfile(inpath,fname), scale_factor, is_negative_stain_data, 1); 
    dims = size(im);
%     figure, imshow(im);
    
    
    cname = ['coordinate_' fname(1:end-4) '_bin3.dat'];
    try
        coordinates = ReadCoordinateTRPV(fullfile(groundtruth_path, cname)); % READ FROM DONATED DATA
    catch
        continue
    end
    coordinates2 = coordinates(:, 3:4);
    coordinates2(:, 1) = coordinates(:, 3);
    coordinates2(:, 2) = dims(1) - coordinates(:, 4);
    
%     cname = [fname(1:end-4) '_autopick.star'];
%     try
%         coordinates = ReadCoordinateStar(fullfile(groundtruth_path, cname)); % READ FROM DONATED DATA
%     catch
%         continue
%     end
%     coordinates2 = coordinates(:, 1:2)/scale_factor;
    
    
    rgb = im;
    numCenters = size(coordinates,1);
    rgb = insertShape(rgb,'Circle',[coordinates2, repmat(rbox_scale, numCenters, 1)],'LineWidth', 3, 'Color', 'Y'); % Already bin 3
%     regcoor2 = [(coordinates2(:,1)-r) (coordinates2(:,2)-r) repmat(2*r, [numCenters 1]) repmat(2*r, [numCenters 1])];
%     rgb = insertShape(rgb, 'rectangle', regcoor2, 'LineWidth', 3, 'Color', 'g');
%     figure, imshow(rgb);
    
    
    cim = zeros(dims);
    cim = insertShape(cim,'FilledCircle',[coordinates2, repmat(floor(rbox_scale/3), numCenters, 1)],'LineWidth', 1, 'Color', 'White'); % Already bin 3
    cim = im2bw(cim);
%     figure, imshow(cim);
    
     
    cim(:,:,1)=imagerange(cim(:,:,1));
%     cim(:,:,2)=imagerange(cim(:,:,2));
%     cim(:,:,3)=uint8(zeros(size(im,1),size(im,1)));

    [rows,cols,channels] = size(im);
    mask_centers = zeros(rows,cols);
    
    R=cim(:,:,1);
    L=bwlabel(double(R));
    stat=regionprops(L,'Centroid');
    xy=[stat.Centroid];
    n=length(xy);
    x=xy(1:2:end);
    y=xy(2:2:end);
    
    ind = sub2ind(size(mask_centers),y,x);
    if (sum(ind<1)>0)
        ind
    end
    mask_centers(round(ind))=1;
    
    
    circles = [x' y' repmat(floor(rbox_scale/3.5), n/2, 1)];
    mask_centers2 = insertShape(mask_centers, 'FilledCircle',  circles, 'LineWidth', 2);
    mask_centers3 = im2bw(mask_centers2, 0.5);
%     figure, imshow(mask_centers3);
    
    %----distance to the center
    d_center=bwdist(mask_centers3);
%     figure, imshow(d_center, []);

    %----distance to boundary
    cells=d_center<floor(rbox_scale/3.5);
    d_border=bwdist(1-cells);
    K=d_border.^2;
%     figure, imshow(K, []);
%     figure, mesh(K);
%     colormap jet;
   
%     rgb=markcontours(im,mask_centers,[0 1 0]);
%     rgb=markcontours(rgb,cells,[0 0 1],[0.5 0.5]);
    

    GT_NEW=cropsquare_up(K,y,x,r_patch);
    original_image=cropsquare_up(im,y,x,r_patch);
    
    
%      mask=d<rad;
    [~, imgnameOR]=fileparts(file_list(i).name);
%     [~, imgnameGT]=fileparts(Flist2(i).name);
    imgnameGT = imgnameOR;
    
    num_patches = size(GT_NEW,4);

    
%     patch_names = cell(num_patches, 1);
%     patch_names = [];

   num_prev = size(patches, 2);


    for k=1:num_patches
        image_name1 = [imgnameGT '_' int2str(k) '.png'];
%         imwrite(uint8(GT_NEW(:,:,:,k)),['GT_bin/' image_name1]);
        
        image_name2 = [imgnameOR '_' int2str(k) '.png'];
%         imwrite(uint8(original_image(:,:,:,k)),['OR_bin/' image_name2]);
        
%         patch_names(k) = {image_name2};
%         patch_names(k) = image_name2;
        patches(num_prev + k).names = image_name2;
        

    end
    
     
     images.data = cat(4, images.data, original_image);
     images.labels = cat(4, images.labels, GT_NEW);
%      images.names = cat(2, images.names, patches.names);

     
     
%      figure, imshow(original_image(:, :, :, 41), []);
%      figure, imshow(GT_NEW(:, :, :, 41), []);
%      
%      figure, imshow(original_image(:, :, :, 66), []);
%      figure, imshow(GT_NEW(:, :, :, 66), []);
     
%      figure, imshow(original_image(:, :, :, end), []);
%      figure, imshow(GT_NEW(:, :, :, end), []);
     
     
    disp([fname '  ' num2str(i)]);

end

%% ======= SAVE CNN Training Files
%  save TRPV_32_64_star.mat images -v7.3; %to save file larger than 2GB
%  save TRPV_32_64_star.mat patches -v7.3; %to save file larger than 2GB
%  save('TRPV_32_64_pre2_88_r9.mat', 'images', 'patches', '-v7.3')
