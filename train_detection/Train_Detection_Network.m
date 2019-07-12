%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
% Program Name: DRPnet Particle Picking
% 
%  Filename: Train_Detection_Network.m
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

clc;

addpath('../source');

%% Set figure for display and save
set(0,'DefaultFigureWindowStyle','docked');
% set(0,'DefaultFigureWindowStyle','normal');

rate = 150; % pixels per inch (DEFAULT)
% set(gcf, 'PaperUnits', 'inches', 'PaperPosition', [0 0 1600 1600]/rate);
% set(gcf, 'units', 'pix', 'pos', [100, 100, 1200, 1200]);

%% Prepare the data

% Load a database of blurred images to train from


% imdb = load('train_detection/TRPV_32_64.mat');
% imdb.images.data(:,:,:,:)=im2double(imdb.images.data(:,:,:,:));
% imdb.images.label=im2double(imdb.images.label);
% imdb.images.data=single(imdb.images.data(:,:,1,:));
% imdb.images.label=single(imdb.images.label);



load('TRPV_32_64_pre2_88_r9.mat');
% 
imdb.images.data = im2double(uint8(images.data));
imdb.images.label = im2double(uint8(images.labels));
imdb.images.data = single(imdb.images.data(:,:,1,:));
imdb.images.label = single(imdb.images.label);




% se=ones(1,10000); se(6000:14000)=2;
% imdb.images.set=se;


%  imdb.images.data=mynormalize_mustd(imdb.images.data);
%  imdb.images.label=mynormalize_mustd(imdb.images.label);


% % Visualize the first image in the database
figure(); set(gcf, 'name', 'Part 3.1: Data') ; clf;

idx = 30;

subplot(1,2,1); imagesc(imdb.images.data(:,:,:,idx));
axis off image ; title('Input (blurred)') ;

subplot(1,2,2); imagesc(uint8(imdb.images.label(:,:,:,idx)));
axis off image ; title('Desired output (sharp)');

% colormap gray ;
% colormap jet;


%% Split Train and Validation Data


N = size(imdb.images.data, 4);
num_data = 1:N;
% BatchSize = N/10;


ratio = [4/5, 1/5, 0];




seed = 0; % Needed to set every call of every random function
rng(seed);
indx = randperm(floor(N/5));
train_id = setdiff(num_data, indx);

valImages = imdb.images.data(:, :, :, indx);
valLabels = imdb.images.label(:, :, :, indx);

trainImages = imdb.images.data(:, :, :, train_id);
trainLabels = imdb.images.label(:, :, :, train_id);

% for k=1:length(indx)
% valImages(:,:,:,k)=trainImages(:,:,:,indx(k));
% valLabels(:,:,:,k)=trainLabels(:,:,:,indx(k));
% end


BatchSize = 16;
train_iter = round(N * ratio(1) / BatchSize);



%% Define network

imageSize = [64 64 1];
convLayers = [
%     imageInputLayer(imageSize);
    imageInputLayer(imageSize, 'Normalization', 'none')
    
    convolution2dLayer([9, 9], 32, 'Padding', 4, 'Name', 'conv_1')
%     batchNormalizationLayer
    reluLayer
    
    convolution2dLayer([3, 3], 32, 'Padding', 1, 'Name', 'conv_2')
%     batchNormalizationLayer
    reluLayer
    
    convolution2dLayer([1, 7], 32,'Padding', [0 0 3 3], 'Name', 'conv_3')
%     batchNormalizationLayer
    reluLayer
    
    convolution2dLayer([7, 1], 32, 'Padding', [3 3 0 0], 'Name', 'conv_4')
%     batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2, 'Stride', 1, 'Padding', [0 1 0 1], 'Name', 'pooling_1')
    
    convolution2dLayer([3, 3], 1, 'Padding', 1, 'Name', 'conv_5')
    
%    RegressionMAELayer('reg_mae_loss')
    RegressionMSELayer('reg_mse_loss')
    ];

%% Training Options

options = trainingOptions('adam',...    
    'MaxEpochs',7,...   
    'MiniBatchSize',BatchSize,...
    'CheckpointPath','check_points',...
    'Shuffle','every-epoch',...
    'InitialLearnRate',0.001,...     
    'LearnRateSchedule','piecewise',...
    'LearnRateDropFactor',0.1,...
    'LearnRateDropPeriod',train_iter,...
    'ValidationData',{valImages,valLabels},...
    'ValidationFrequency',train_iter,...
    'VerboseFrequency',train_iter,...
'Plots','training-progress');


%% Train network

% cnet = trainNetwork(imdb.images.data, imdb.images.label, convLayers, options);
cnet = trainNetwork(trainImages, trainLabels, convLayers, options);

% save('cnet_1_new.mat', 'cnet');

%% Select particle to test
idx2 = 12;
val_im = imdb.images.data(:, :, :, idx2);
val_lb = imdb.images.label(:, :, :, idx2);

f1 = figure, imshow(val_im , []);
f2 = figure, imshow(val_lb, []); colormap gray;

f3 = figure, imshow(val_lb, [], 'colormap', jet); 
% colorbar;
f4 = figure, mesh(val_lb);


%% Test to predict particle's map
YPred = predict(cnet, val_im);
YPred = imgaussfilt(YPred, 8);

f5 = figure, imshow(YPred, []); colormap gray;
f6 = figure, imshow(YPred, [], 'colormap', jet); colorbar;
f7 = figure, mesh(YPred);




