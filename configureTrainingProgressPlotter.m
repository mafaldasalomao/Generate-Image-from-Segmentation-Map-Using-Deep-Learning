function lossPlotter = configureTrainingProgressPlotter(f)
% The configureTrainingProgressPlotter function configures training
% progress plots for various losses.

figure(f);
clf
subplot(2,2,1)
xlabel('Iteration');
ylabel('Discriminator Loss');
lossPlotter.dline = animatedline;

subplot(2,2,2);
xlabel('Iteration');
ylabel('Generator Loss');
lossPlotter.gline = animatedline;

subplot(2,2,3);
xlabel('Iteration');
ylabel('Feature Matching Loss');
lossPlotter.fmline = animatedline;
subplot(2,2,4);
xlabel('Iteration');
ylabel('VGG Loss');
lossPlotter.vggline = animatedline;

end