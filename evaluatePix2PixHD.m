function [generatedImage,segMap] = evaluatePix2PixHD(pxds,idx,imageSize,net)
% The evaluatePix2PixHD function predicts the generated image
% using trained generator for given pixel-labeled input image. The
% function performs necessary and pre- and post-processing steps to
% process categorical input and predicted uint8 output.

% Get the pixel label image from the test data. Resize the pixel label image
segMap = readimage(pxds,idx);
segMap = imresize(segMap,imageSize,"nearest");

% Convert the pixel label image to a multichannel one-hot image
segMapOneHot = onehotencode(segMap, 3, 'single');

% Convert data to dlarray specify the dimension labels 'SSCB'
% (spatial, spatial, channel, batch)
dlSegMap = dlarray(segMapOneHot,'SSCB');

% If training on a GPU, then convert data to gpuArray
if canUseGPU
    dlSegMap = gpuArray(dlSegMap);
end

% Generate the image
generatedImage = predict(net,dlSegMap);
generatedImage = extractdata(gather(generatedImage));

% Rescale the image to the range [0, 1]
generatedImage = rescale(generatedImage);
    
end