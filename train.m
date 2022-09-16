%Train the Network

% Read the data for current mini-batch using the next (Deep Learning Toolbox) function.
% Evaluate the model gradients using the dlfeval (Deep Learning Toolbox) function and the modelGradients helper function.
% Update the network parameters using the adamupdate (Deep Learning Toolbox) function.
% Update the training progress plot for every iteration and display various computed losses.

doTraining = false;
if doTraining
    fig = figure;    
    
    lossPlotter = configureTrainingProgressPlotter(fig);
    iteration = 0;

    % Loop over epochs
    for epoch = 1:numEpochs
        
        % Reset and shuffle the data
        reset(mbqTrain);
        shuffle(mbqTrain);
 
        % Loop over each image
        while hasdata(mbqTrain)
            iteration = iteration + 1;
            
            % Read data from current mini-batch
            [dlInputSegMap,dlRealImage] = next(mbqTrain);
            
            % Evaluate the model gradients and the generator state using
            % dlfeval and the GANLoss function listed at the end of the
            % example
            [gradParamsG,gradParamsDScale1,gradParamsDScale2,lossGGAN,lossGFM,lossGVGG,lossD] = dlfeval( ...
                @modelGradients,dlInputSegMap,dlRealImage,dlnetGenerator,dlnetDiscriminatorScale1,dlnetDiscriminatorScale2,netVGG);
            
            % Update the generator parameters
            [dlnetGenerator,trailingAvgGenerator,trailingAvgSqGenerator] = adamupdate( ...
                dlnetGenerator,gradParamsG, ...
                trailingAvgGenerator,trailingAvgSqGenerator,iteration, ...
                learningRate,gradientDecayFactor,squaredGradientDecayFactor);
            
            % Update the discriminator scale1 parameters
            [dlnetDiscriminatorScale1,trailingAvgDiscriminatorScale1,trailingAvgSqDiscriminatorScale1] = adamupdate( ...
                dlnetDiscriminatorScale1,gradParamsDScale1, ...
                trailingAvgDiscriminatorScale1,trailingAvgSqDiscriminatorScale1,iteration, ...
                learningRate,gradientDecayFactor,squaredGradientDecayFactor);
            
            % Update the discriminator scale2 parameters
            [dlnetDiscriminatorScale2,trailingAvgDiscriminatorScale2,trailingAvgSqDiscriminatorScale2] = adamupdate( ...
                dlnetDiscriminatorScale2,gradParamsDScale2, ...
                trailingAvgDiscriminatorScale2,trailingAvgSqDiscriminatorScale2,iteration, ...
                learningRate,gradientDecayFactor,squaredGradientDecayFactor);
            
            % Plot and display various losses
            lossPlotter = updateTrainingProgressPlotter(lossPlotter,iteration, ...
                epoch,numEpochs,lossD,lossGGAN,lossGFM,lossGVGG);
        end
    end
    save('trainedPix2PixHDNet.mat','dlnetGenerator');
    
else    
    trainedPix2PixHDNet_url = 'https://ssd.mathworks.com/supportfiles/vision/data/trainedPix2PixHDv2.zip';
    netDir = fullfile(tempdir,'CamVid');
    downloadTrainedPix2PixHDNet(trainedPix2PixHDNet_url,netDir);
    load(fullfile(netDir,'trainedPix2PixHDv2.mat'));
end