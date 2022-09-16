function [imdsTrain,imdsTest,pxdsTrain,pxdsTest] = partitionCamVidForPix2PixHD(imds,pxds,classes,labelIDs)
% The partitionCamVidForPix2PixHD function partitions the CamVid data set
% into training and testing data sets. 92.5% of the data is used for
% training and the rest is used for testing.
    
% Set the initial random state for example reproducibility.
rng('default');
numFiles = length(imds.Files);
shuffledIndices = randperm(numFiles);

% Use the first 92.5% of files for training.
trainingDataRatio = 0.925;
numTrain = round(trainingDataRatio * numFiles);
trainingIdx = shuffledIndices(1:numTrain);

% Use the rest of the files for testing.
testIdx = shuffledIndices(numTrain+1:end);

% Create image datastores containing ground truth scene images for training
% and testing.
trainingImages = imds.Files(trainingIdx);
testImages = imds.Files(testIdx);

imdsTrain = imageDatastore(trainingImages);
imdsTest = imageDatastore(testImages);

% Create pixel label datastores containing pixel labels for training and
% testing.
trainingLabels = pxds.Files(trainingIdx);
testLabels = pxds.Files(testIdx);
pxdsTrain = pixelLabelDatastore(trainingLabels,classes,labelIDs);
pxdsTest = pixelLabelDatastore(testLabels,classes,labelIDs);
end