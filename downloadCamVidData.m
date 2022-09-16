function downloadCamVidData(dataDir,imageURL,labelURL)
% The downloadCamVidData function downloads the CamVid dataset and saves it
% into the directory specified by dataDir.

if ~exist(dataDir,'dir')   
    mkdir(dataDir)
    labelsZip = fullfile(dataDir,'labels.zip');
    imagesZip = fullfile(dataDir,'images.zip');   
    
    disp('Downloading 16 MB CamVid data set labels...'); 
    websave(labelsZip,labelURL);
    unzip(labelsZip,fullfile(dataDir,'labels'));
    fprintf('Done.\n\n');
    
    disp('Downloading 557 MB CamVid data set images...');  
    websave(imagesZip,imageURL);       
    unzip(imagesZip,fullfile(dataDir,'images')); 
    fprintf('Done.\n\n');
end
end