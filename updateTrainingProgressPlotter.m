function lossPlotter = updateTrainingProgressPlotter(lossPlotter,iteration,epoch,numEpochs,lossD,lossGGAN,lossGFM,lossGVGG)
% The updateTrainingProgressPlotter function updates plots for every
% iteration with new losses.

addpoints(lossPlotter.dline,iteration,double(lossD)); 
addpoints(lossPlotter.gline,iteration,double(lossGGAN));
addpoints(lossPlotter.fmline,iteration,double(lossGFM));
addpoints(lossPlotter.vggline,iteration,double(lossGVGG));
 
sgtitle("Training Epoch: " + epoch + " of " + numEpochs);
drawnow

end 