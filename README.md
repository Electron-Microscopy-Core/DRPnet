
# DRPnet - Automated Particle Picking In Cryo-Electron Micrographs Using Deep Regression

This repository is an implementation in Matlab for the paper [DRPnet - Automated Particle Picking In Cryo-Electron Micrographs Using Deep Regression](https://www.biorxiv.org/content/10.1101/616169v1).


## Prerequisites
The code was tested successfully on the system of:  
Matlab 2018b (/w Image Processing Toolbox, Computer Vision Toolbox, and Deep Learning Toolbox)   
CUDA 9.2   
cuDNN 7.4 


## How to pick particles
### Setting up the program parameters:
The program parameters are stored in file "Input/file_name.txt". To pick particles from a dataset of images, we need to set these parameters to appropriate values. There are seven groups of parameters we can change:

```
% ------------ input/output paths ---------------------------------------------------------
inpath = 'path_to_your_images'; % folder contains .mrc files
outpath = 'output'; % folder stores coordinate files (.star files) of picked particles
groundruth_path = ''; % folder stores ground truth coordinates files

% ------------- train/test data -----------------------------------------------------------
start_train_detect = 2;  % index of the first micrograph to train detection 
num_train_detect = 2;     % number of images to train classification network
start_test_detect = 2;   % index of the first micrograph to test detection (and train/test classification)
num_train_classify = 2;   % number of images to train classification network

% ------------- input image type: negative stain versus cryo-em
is_negative_stain_data = 1;

% ------------ prticle diameter in pixels -------------------------------------------------
box_size = 50;

% ------------ peak detection parameters, used after DRPnet -------------------------------
sigma_detect = 3;
threshold = 8;
k_level = 3;

%------------- classification network (CNN-2) parameters ----------------------------------
retrain = 0; 
num_train_mics = 2;
num_epochs = 5;
class_cutoff = 0.7;

%------------- optional: filter/don't filter particles based on their intensity -----------
customize_pct_training = 0;
pct_mean_training = 1;
pct_std_training = 50;

customize_pct_predict = 1;
pct_mean_predict = 1;
pct_std_predict = 50;

filter_particle = 1;
-------------------------------------------------------------------------------------------
```

## Run picking program
```
matlab -r -nosplash -nodesktop DRPnet('path/file_name')
```
The coordinate files of pick particles will be generated in "output" folder

## How to retrain classification network (CNN-2)
Users can set the retrain paramter to 1 if they need to train the classification network (CNN-2) to adapt better with a specific type of particles.
```
%------------- classification network (CNN-2) parameters ----------------------------------
retrain = 1; 
```

Then run the program 
```
matlab -r -nosplash -nodesktop DRPnet('path/file_name')
```
or open the file "Test_Detection_Classification_DRPnet.m" in Matlab to run in interactive mode.

At first this program detects postive samples and negative samples, and trains the CNN-2 network. After finishing, the program restarts its iteration to detect particles, and will use the recenlty trained classification network to classify true/false instances of particles, and save them in the "output" folder.

## How to retrain detection network (CNN-1)
The pre-trained Fully Convolutional Regression Network (FCRN) works well with multiple types of particles based on blob detection method. Users can also train FCRN network by their own using the files provided in "train_detection" folder.
First, users can prepare the training samples by running

```
matlab GetTrainingDetectionSamples_bin.m
```
Then users begin to train FCRN network (CNN-1):
```
matlab Train_Detection_Network.m
```

