%generate a synthetic image of a scene from a semantic segmentation map
% using a pix2pixHD conditional generative adversarial network (CGAN).
%Pix2pixHD [1] consists of two networks that are trained simultaneously to
% maximize the performance of both

%The generator is an encoder-decoder style neural network that generates a 
% scene image from a semantic segmentation map
clear all, close all;


%% Download Dataset
imageURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip';
labelURL = 'http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip';

dataDir = fullfile(tempdir,'CamVid'); 
downloadCamVidData(dataDir,imageURL,labelURL);
imgDir = fullfile(dataDir,"images","701_StillsRaw_full");
labelDir = fullfile(dataDir,'labels');
%% 

imds = imageDatastore(imgDir);
imageSize = [576 768];
%% Create Pixels id
numClasses = 32;
[classes,labelIDs] = defineCamVid32ClassesAndPixelLabelIDs;
cmap = camvid32ColorMap;
%Create a pixelLabelDatastore to store the pixel label images.
pxds = pixelLabelDatastore(labelDir,classes,labelIDs);

%display
im = preview(imds);
px = preview(pxds);
px = label2rgb(px,cmap);
montage({px,im})

%% Pratition data
[imdsTrain,imdsTest,pxdsTrain,pxdsTest] = partitionCamVidForPix2PixHD(imds, ...
    pxds,classes,labelIDs);

dsTrain = combine(pxdsTrain,imdsTrain);

%% 576-by-768 pixels

%Scale the ground truth data to the range [-1, 1]. This range matches the range of the final tanhLayer (Deep Learning Toolbox) in the generator network.
    %Hyperbolic tangent (tanh) layer
%Resize the image and labels to the output size of the network, 576-by-768 pixels, using bicubic and nearest neighbor downsampling, respectively.
%Convert the single channel segmentation map to a 32-channel one-hot encoded segmentation map using the onehotencode (Deep Learning Toolbox) function.
%Randomly flip image and pixel label pairs in the horizontal direction.


dsTrain = transform(dsTrain,@(x) preprocessCamVidForPix2PixHD(x,imageSize));

%% Create Generator Network

generatorInputSize = [imageSize numClasses];
%Create the pix2pixHD generator network using the pix2pixHDGlobalGenerator function.
dlnetGenerator = pix2pixHDGlobalGenerator(generatorInputSize);

% Display the network architecture.
%analyzeNetwork(dlnetGenerator)


%% Create Discriminator Network