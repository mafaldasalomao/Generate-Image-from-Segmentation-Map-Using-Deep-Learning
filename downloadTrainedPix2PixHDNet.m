function downloadTrainedPix2PixHDNet(url,destination)
% The downloadTrainedPix2PixHDNet function downloads a pretrained Pix2PixHD
% network that generates synthetic scene images from pixel label images.
%
% Copyright 2021 The MathWorks, Inc.

[~,name,~] = fileparts(url);
netDirFullPath = destination;
netZipFileFullPath = fullfile(destination,[name '.zip']);
netMATFileFullPath = fullfile(destination,[name '.mat']);
if ~exist(netMATFileFullPath,'file')
    if ~exist(netZipFileFullPath,'file')
        fprintf('Downloading pretrained Pix2PixHD network for CamVid data.\n');
        fprintf('This can take several minutes to download...\n');
        if ~exist(netDirFullPath,'dir')
            mkdir(netDirFullPath);
        end
        websave(netZipFileFullPath,url);
        fprintf('Done.\n\n');
    end
    unzip(netZipFileFullPath,netDirFullPath)
else
    fprintf('Pretrained Pix2PixHD network for CamVid data set already exists.\n\n');
end
end