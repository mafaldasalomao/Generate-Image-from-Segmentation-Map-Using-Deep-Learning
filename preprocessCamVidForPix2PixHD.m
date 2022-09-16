function transformedData = preprocessCamVidForPix2PixHD(inputData,imageSize)
% The preprocessCamVidForPix2PixHD function resizes the input to the
% network input size and rescales image pixels values to the range [-1, 1].
% The function converts each pixel label image to a multichannel one-hot
% encoded image. Each channel represents the segmentation of
% a unique class, in which the class has the value 1 and all other classes
% have the value 0.

% Process a pixel label image and the corresponding ground truth scene
% image.
segMap = inputData{1,1};
realImage = inputData{1,2};

realImage = im2single(realImage);
realImage = (realImage - 0.5)/0.5;
realImage = imresize(realImage,imageSize,"bicubic");

% Convert the pixel label image to a multichannel one-hot encoded image.
segMap = imresize(segMap,imageSize,"nearest");
segMap = onehotencode(segMap,3,'single');
% Convert the NaNs in label to 0.
segMap(isnan(segMap)) = 0;

% Augment the data set by randomly applying horizontal reflections to
% the scene image and pixel label image pairs.
if rand > 0.5
    segMap = fliplr(segMap);
    realImage = fliplr(realImage);
end
transformedData{1,1} = segMap;
transformedData{1,2} = realImage;

end