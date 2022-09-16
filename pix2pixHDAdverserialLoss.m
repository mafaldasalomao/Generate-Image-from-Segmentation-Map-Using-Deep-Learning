function [DLoss,GLoss,realPredFtrsD,genPredFtrsD] = pix2pixHDAdverserialLoss(inpReal,inpGenerated,discriminator)
    %Adversarial Loss Function
    % Discriminator layer names containing feature maps
    featureNames = {'act_top','act_mid_1','act_mid_2','act_tail','conv2d_final'};
    
    % Get the feature maps for the real image from the discriminator    
    realPredFtrsD = cell(size(featureNames));
    [realPredFtrsD{:}] = forward(discriminator,inpReal,"Outputs",featureNames);
    
    % Get the feature maps for the generated image from the discriminator    
    genPredFtrsD = cell(size(featureNames));
    [genPredFtrsD{:}] = forward(discriminator,inpGenerated,"Outputs",featureNames);
    
    % Get the feature map from the final layer to compute the loss
    realPredD = realPredFtrsD{end};
    genPredD = genPredFtrsD{end};
    
    % Compute the discriminator loss
    DLoss = (1 - realPredD).^2 + (genPredD).^2;
    DLoss = mean(DLoss,"all");
    
    % Compute the generator loss
    GLoss = (1 - genPredD).^2;
    GLoss = mean(GLoss,"all");
end