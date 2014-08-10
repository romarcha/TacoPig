%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                     Gaussian Process Demo Script
%  Demonstrates GP regression using the taco-pig toolbox on 1-D Data.
%
%  Note: when the size of the original dataset increases  importantly
%        (more than 10000), the convergence of the hyperparameters becomes
%        harder.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Add optimization folder
if ~exist('minfunc')
    fprintf('It looks like you need to add minfunc to your default path...\n');
    fprintf('(Add tacopig/optimization/minfunc{/mex} to pathdef.m for permanent access)\n');
    fprintf('Press any key to attempt to continue...\n');
    pause();
end
tacopigroot = which('tacopig.taco');
tacopigroot = tacopigroot(1:end-15);
addpath(genpath([tacopigroot,'optimization']))

%% %%%%%%%%%%%%%% 1-D Example of Subset of Regressors%%%%%%%%%%%%%%%%%%%%% 
close all; clear functions; clc;

%% Set up 1-D Data
% Training Data
% 8000 Noisy samples from an unknown function.
n_samples = 8000;
noise_std = 0.1;
unknown_function = @(x)0.5*(sin(17*(x+0.3)).*((x+0.3).^(1/2))-0.7*cos(30*x));
X = random('unif',0,1,[1, n_samples]);
noise = random('Normal',0,noise_std,1,size(X,2));
y = feval(unknown_function,X)+noise;

n_induced = 40;

n = size(X,2);
[X id] = sort(X);
y = y(id);

try % pick the induced points. only half as many points in this case.
    [indxs, induced] = kmeans(X, n_induced);
catch
%     induced = (rand(10,1)-0.5) *abs(max(X)-min(X))+mean(X) ;
    induced = [linspace(min(X),max(X),n_induced)]';
end
% we will now compute the regression over these induced points


%% Set up Gaussian process

% Use a standard GP regression model:
GP = tacopig.gp.SubsetRegressor;

% Plug in the data
GP.X = X;
GP.XI = induced';
GP.y = y;

% Plug in the components
GP.MeanFn  = tacopig.meanfn.FixedMean(mean(y));
GP.CovFn   = tacopig.covfn.SqExp();
GP.NoiseFn = tacopig.noisefn.Stationary();
GP.objective_function = @tacopig.objectivefn.SR_LMLG;

% GP.solver_function = @anneal;

% Initialise the hyperparameters
GP.covpar   = 2*ones(1,GP.CovFn.npar(size(X,1)));
GP.meanpar  = zeros(1,GP.MeanFn.npar(size(X,1)));
GP.noisepar = 0.4*ones(1,GP.NoiseFn.npar);


%% Before Learning: Query
GP.solve(); 
xstar = linspace(min(X), max(X), 201); 
[mf, vf] = GP.query(xstar);
sf  = sqrt(vf);

% Display predicitve mean and variance
figure
plot(X, y, 'k+', 'MarkerSize', 17)
f  = [mf+2*sf,flipdim(mf-2*sf,2)]';
h(1) = fill([xstar, flipdim(xstar,2)], f, [6 6 6]/8, 'EdgeColor', [6 6 6]/8);
hold on
h(2) = plot(xstar,mf,'k-','LineWidth',2);
h(3) = plot(X, y, 'k+', 'MarkerSize', 17);
h(4) = plot(induced', interp1(xstar,mf,induced'), 'ro');
title('Before Hyperparameter Training');
legend(h,'Predictive Standard Deviation','Predictive Mean', 'Training Points','Induced Points','Location','SouthWest')

%% Learn & Query
GP.learn();
GP.solve();
[mf, vf] = GP.query(xstar);
sf  = sqrt(vf);

% Display learnt model
figure
plot(X, y, 'k+', 'MarkerSize', 17)
f  = [mf+2*(sf),flipdim(mf-2*(sf),2)]';
h(1) = fill([xstar, flipdim(xstar,2)], f, [6 6 6]/8, 'EdgeColor', [6 6 6]/8);
hold on
h(2) = plot(xstar,mf,'k-','LineWidth',2);
h(3) = plot(X, y, 'k+', 'MarkerSize', 17);
h(4) = plot(induced', interp1(xstar,mf,induced'), 'ro');
title('After Hyperparameter Training');
legend(h,'Predictive Standard Deviation','Predictive Mean', 'Training Points','Induced Points','Location','SouthWest')
