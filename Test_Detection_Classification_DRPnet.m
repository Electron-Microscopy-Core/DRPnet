%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% Program Name: DRPnet Particle Picking
%
%  Filename: Test_Detection_Classification_DRPnet.m
%
%  Description: Perform particle detection, and then classification with or
%  without re-training classification network (CNN-2)
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

%%
set(0,'DefaultFigureWindowStyle','docked');
close all;
clear all;
clc;

addpath('source');

%% SET UP PARAMETERS

input_file = 'input/10061_params_TEST.txt';
% input_file = 'input/10017_params_TEST.txt';
% input_file = 'input/10005_params_TEST.txt';

input_file = 'input/OKT3_params_TRAIN_TEST.txt';
% input_file = 'input/10061_params_TRAIN_TEST.txt';
% input_file = 'input/10017_params_TRAIN_TEST.txt';
% input_file = 'input/10005_params_TRAIN_TEST.txt';

[names vals inputParams]=ParseParams(input_file);

for i=1:length(names)
    if isnumeric(vals{i})
        eval([genvarname(names{i}) ' = ' num2str(vals{i}) ';']);
    else
        eval([genvarname(names{i}) ' = ' vals{i} ';']);
    end
end

[scale_factor, rbox_scale, sigma_gauss, f3] = SetScaleFactors(inpath, box_size);


%% LOAD DEEP LEARNING NETWORK
% Load Detection Network
convLayers = CreateDetectionNet();
load('models/cnet_1.mat');

% Load Classification Network
load('models/cnet_2.mat');

%% DETECT AND CLASSIFY PARTICLE
training = retrain;
linewidth = 2;

patches_pos = [];
patches_neg = [];
store_struct = [];

flist=dir(fullfile(inpath,'*.mrc'));
num_images=length(flist);
index = start_test_detect;
last_train_index = index + num_train_classify - 1;







while index <= num_images
    
    % ============================= PRE-PROCESS ===========================
    fname = flist(index).name;
    store_struct(index).FileName = fname;
    im = PreprocessImage(fullfile(inpath,fname), scale_factor, is_negative_stain_data, 0);
    imageDims = size(im);
    figure, imshow(im);
    originImage = im;
    
    
    % ======================== DETECT PARTICLES ===========================
    [clist centers2 particleMeans_org particleStd_org particlePatches_org] = DetectParticles(im, cnet, rbox_scale, sigma_gauss, f3, sigma_detect, threshold, k_level);
    K=size(clist, 2);
    numCenters2 = size(centers2, 1);
    
    % --- Visualize detection --
    rgb=im;
    mrcFig = figure;
    imshow(rgb);
    hold on;
    % figure, imshow(rgb, 'colormap', jet); hold on;
    % figure, imshow(rgb, 'colormap', gray); hold on;
    
    imageParticleBoxes = [];
    patchSize = [rbox_scale*2 rbox_scale*2];
    colors=[1 0 0; 1 1 0; 0 0 1];
    
    for k=3:K
        centers=clist(k).centers;
        figure(mrcFig);
        plot(centers(:,1),centers(:,2),'o','Color',colors(k,:),'LineWidth',linewidth/2,'MarkerSize', rbox_scale/2);
        % Compare2GT_centers(GT(:,:,1),centers,Tabs,Trel,method,fr,evalpath,fname);
    end
    
    
    % =================== RE-TRAIN CLASSIFICATION NETWORK =================
    if training == 1
        
        % --- Prepare training samples ---
        if customize_pct_training == 0
            pct_mean_training = 2;
            pct_std_training = 20;
        end
        [patches_pos patches_neg mask2_pos centers_neg2 p_array_train] = GetTrainingClassificationSamples(patches_pos, patches_neg, ...
                                                                                                im, rbox_scale, k_level, clist, centers2, ...
                                                                                                particleMeans_org, particleStd_org, particlePatches_org, ...
                                                                                                pct_mean_training, pct_std_training);
        
        % --- Visualize training samples ---
        
        figure(mrcFig);
        for j = 1:numCenters2
            if (mask2_pos(j) == 0)
                text(centers2(j,1), centers2(j,2), sprintf('%d', j), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'red');
            else
                text(centers2(j,1), centers2(j,2), sprintf('%d', j), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'blue');
            end
        end
        
        num_neg_mic = size(centers_neg2, 1);
        plot(centers_neg2(:,1), centers_neg2(:,2), 'o', 'Color', 'magenta', 'LineWidth', linewidth/2, 'MarkerSize', rbox_scale/2);
        for j = 1:num_neg_mic
            text(centers_neg2(j,1), centers_neg2(j,2), sprintf('%d', j), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'magenta');
        end
        
        
        % --- Re-train the classification network ---
        if index == last_train_index
            num_pos = size(patches_pos, 4);
            num_neg = size(patches_neg, 4);
            num_class_min = min(num_pos, num_neg);
            trainX = cat(4, patches_pos(:, :, 1, 1:num_class_min), patches_neg(:, :, 1, 1:num_class_min));
            trainY = zeros(2*num_class_min, 1);
            trainY(1:num_class_min, :) = 1;
            trainY = categorical(trainY);
            
            net2 = TrainClassificationNet(trainX, trainY, num_epochs);
            
            training = 0;
            index = start_test_detect;
            continue
        else
            index = index + 1;
            continue
        end
        
        
    end
    
    
    
    % ========================= CLASSIFICATION ============================
    if retrain == 1
        net_test = net2;
    else
        net_test = net;
    end
    
    if customize_pct_training == 0
        pct_mean_predict = 1;
        pct_std_predict = 10;
    end
    [mask2 prob filterList_screen p_array_test] = ClassifyPrediction(net_test, class_cutoff, centers2, ...
                               particleMeans_org, particleStd_org, particlePatches_org, ...
                               filter_particle, pct_mean_predict, pct_std_predict);
    
    % Classification results
    centers3 = centers2(mask2, :);
    particlePatches_org3 =  particlePatches_org(:, :, mask2);
    
    % -------------- Visualization ----------------------------------------
    figure(mrcFig);
    particleCount = size(centers2, 1);
    for j = 1:particleCount
        if (mask2(j) == 0) 
            text(centers2(j,1), centers2(j,2), sprintf('%d', j), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'red');
        else
            text(centers2(j,1), centers2(j,2), sprintf('%d', j), 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', 'Color', 'blue');
        end
    end
    
    
    % ===================== GET PICKED PARTICLES ==========================
    % --- Show picked particles ---
    viz_im = im;
    if rbox_scale > 32
        rbox_viz = rbox_scale * 2/3;
    else
        rbox_viz = rbox_scale;
    end
    viz_im = insertShape(viz_im, 'circle', [centers3 repmat(rbox_viz, size(centers3, 1), 1)], 'LineWidth', 4, 'Color', 'y');
    figure, imshow(viz_im);
    fname6=strrep(fname,'.png', '_centers3.png');
    % imwrite(viz_im, fullfile('results', fname6));
    
    % --- Write picked particles' coordinates ---
    WriteStarFile(fname, outpath, centers3, 'auto');
    
    disp([fname ' ' num2str(index)]);
    
    
    
    
    % ========================== NEXT IMAGE ===============================
    index = index + 1;
    
    
end
store_table = struct2table(store_struct);






