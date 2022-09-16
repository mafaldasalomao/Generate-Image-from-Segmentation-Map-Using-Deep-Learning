idxToTest = 1;
gtImage = readimage(imdsTest,idxToTest);
gtImage = imresize(gtImage,imageSize,"bicubic");

segMap = readimage(pxdsTest,idxToTest);
segMap = imresize(segMap,imageSize,"nearest");

segMapOneHot = onehotencode(segMap,3,'single');

dlSegMap = dlarray(segMapOneHot,'SSCB'); 
if canUseGPU
    dlSegMap = gpuArray(dlSegMap);
end

%Generate a scene image from the generator and one-hot segmentation map using the predict (Deep Learning Toolbox) function.

dlGeneratedImage = predict(dlnetGenerator,dlSegMap);
generatedImage = extractdata(gather(dlGeneratedImage));
generatedImage = rescale(generatedImage);
coloredSegMap = label2rgb(segMap,cmap);

figure
montage({coloredSegMap generatedImage gtImage},'Size',[1 3])
title(['Test Pixel Label Image ',num2str(idxToTest),' with Generated and Ground Truth Scene Images'])

idxToTest = 3;  
gtImage = readimage(imdsTest,idxToTest);
gtImage = imresize(gtImage,imageSize,"bicubic");

[generatedImage,segMap] = evaluatePix2PixHD(pxdsTest,idxToTest,imageSize,dlnetGenerator);

coloredSegMap = label2rgb(segMap,cmap);


figure
montage({coloredSegMap generatedImage gtImage},'Size',[1 3])
title(['Test Pixel Label Image ',num2str(idxToTest),' with Generated and Ground Truth Scene Images'])


%% Evaluate Generated Images from Custom Pixel Label Images
cpxds = pixelLabelDatastore(pwd,classes,labelIDs);
for idx = 1:length(cpxds.Files)

    % Get the pixel label image and generated scene image
    [generatedImage,segMap] = evaluatePix2PixHD(cpxds,idx,imageSize,dlnetGenerator);
    
    % For display, convert the labels from categorical labels to RGB colors
    coloredSegMap = label2rgb(segMap);
    
    % Display the pixel label image and generated scene image in a montage
    figure
    montage({coloredSegMap generatedImage})
    title(['Custom Pixel Label Image ',num2str(idx),' and Generated Scene Image'])

end
