%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                       Gaussian Process Demo Script
%  Demonstrates GP regression using the taco-pig toolbox on 1-D Data.
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


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% 1-D Example%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
close all; clear functions; clc;

%% Set up 1-D Data

%% Ground truth of the goal function

%%
% Algebraic expression of the goal function. (Unknown for the algorithm)
clear goalFunction;
goalFunction = GoalFunction();
goalFunction.expression = @(x)0.5*(sin(17*(x+0.3)).*((x+0.3).^(1/2))-0.7*cos(30*x));


%%
% Set the limits of the goal function
goalFunction.limits.low = 0;
goalFunction.limits.high = 1;

%%
% Set the gaussian noise parameters when sampling the function
goalFunction.noiseMean = 0;
goalFunction.noiseStd = 0.1;

%%
% Plot of the goal function values
numberOfSamples = 100;
%figure;
%goalFunction.Plot(numberOfSamples);

%% Learn the GP model hyperparameters
% As it is stated in the state of the art, the function is randomnly
% sampled with "enough" samples to learn the hyperparameters.
% (Something that is not completely correct for me). However
% in the robotics field it can be understood as some random movement
% over the environment.

%%
% Sample the function randomly at "enough" points

%%
% Number of sample points
N = 1000;

%%
% Location of sample points
samplePoints = goalFunction.GetRandomEvaluationLocations(N);

%%
% Get Noisy samples
sampleValues = goalFunction.GetNoisySample(samplePoints);

%%
% Plot original ground truth and sample values with noise
figure;
goalFunction.Plot(numberOfSamples)
hold on;
plot(samplePoints,sampleValues,'rx');
title(sprintf('Sample values with noise mean %.3f and std %.3f',goalFunction.noiseMean,goalFunction.noiseStd));

xstar = goalFunction.GetEvaluationDomain(200);

% Order data points for visualisation
X = samplePoints';
y = sampleValues';
[X id] = sort(X);
y = y(id);
X(200:300) = [];
y(200:300) = [];
X(500:600) = [];
y(500:600) = [];

%% Full Gaussian process
% Use a standard GP regression model:
FullGP = tacopig.gp.Regressor;

% Plug in the data
FullGP.X = X;
FullGP.y = y;

% Plug in the components
FullGP.MeanFn  = tacopig.meanfn.FixedMean(mean(y));
% GP.CovFn   = tacopig.covfn.Sum(tacopig.covfn.Mat3(),tacopig.covfn.SqExp());%SqExp();
FullGP.CovFn   = tacopig.covfn.SqExp();
FullGP.NoiseFn = tacopig.noisefn.Stationary();
% GP.objective_function = @tacopig.objectivefn.CrossVal;
FullGP.objective_function = @tacopig.objectivefn.NLML;
FullGP.opts.numDiff = 1; %Derivatives are calculated numerically
% GP.solver_function = @anneal;

% Initialise the hyperparameters
FullGP.covpar   = 0.5*ones(1,FullGP.CovFn.npar(size(X,1)));
FullGP.meanpar  = zeros(1,FullGP.MeanFn.npar(size(X,1)));
FullGP.noisepar = 1e-1*ones(1,FullGP.NoiseFn.npar);

%% Learn & Query
tic;
FullGP.learn();
FullGPTime = toc();
FullGP.solve();
[mf_full, vf_full] = FullGP.query(xstar);
sf_full  = sqrt(vf_full);

%% Display learnt model
figure
f  = [mf_full+2*(sf_full),flipdim(mf_full-2*(sf_full),2)]';
h(1) = fill([xstar, flipdim(xstar,2)], f, [6 6 6]/8, 'EdgeColor', [6 6 6]/8);
hold on
h(2) = plot(xstar,mf_full,'k-','LineWidth',2);
h(3) = plot(X, y, 'k+');
title('After Hyperparameter Training');
legend(h,'Predictive Standard Deviation','Predictive Mean', 'Training Points','Location','SouthWest')

%% Set up Sparse Gaussian process

% Use a standard GP regression model:
GP = tacopig.gp.Regressor;

% Plug in the data
GP.X = X;
GP.y = y;

% Plug in the components
GP.MeanFn  = tacopig.meanfn.FixedMean(mean(y));
% GP.CovFn   = tacopig.covfn.Sum(tacopig.covfn.Mat3(),tacopig.covfn.SqExp());%SqExp();
GP.CovFn   = tacopig.covfn.Sparse(tacopig.covfn.SqExp(),0.01,1);
GP.NoiseFn = tacopig.noisefn.Stationary();
% GP.objective_function = @tacopig.objectivefn.CrossVal;
GP.objective_function = @tacopig.objectivefn.NLML;
GP.opts.numDiff = 1; %Derivatives are calculated numerically
% GP.solver_function = @anneal;

% Initialise the hyperparameters
GP.covpar   = 0.5*ones(1,GP.CovFn.npar(size(X,1)));
GP.meanpar  = zeros(1,GP.MeanFn.npar(size(X,1)));
GP.noisepar = 1e-1*ones(1,GP.NoiseFn.npar);

%% Learn & Query
tic;
GP.learn();
GPTime = toc();
GP.solve();
[mf, vf] = GP.query(xstar);
sf  = sqrt(vf);

% Display learnt model
figure
f  = [mf+2*(sf),flipdim(mf-2*(sf),2)]';
h(1) = fill([xstar, flipdim(xstar,2)], f, [6 6 6]/8, 'EdgeColor', [6 6 6]/8);
hold on
h(2) = plot(xstar,mf,'k-','LineWidth',2);
h(3) = plot(X, y, 'k+');
title('After Hyperparameter Training');
legend(h,'Predictive Standard Deviation','Predictive Mean', 'Training Points','Location','SouthWest')

%% Compare
figure
f  = [mf+2*(sf),flipdim(mf-2*(sf),2)]';
f_full  = [mf_full+2*(sf_full),flipdim(mf_full-2*(sf_full),2)]';
h(1) = fill([xstar, flipdim(xstar,2)], f, [7 7 7]/8, 'EdgeColor', [7 7 7]/8);
hold on
h(2) = fill([xstar, flipdim(xstar,2)], f_full, [6 6 6]/8, 'EdgeColor', [6 6 6]/8);
goalFunction.Plot(numberOfSamples);
h(3) = plot(xstar,mf,'k-','LineWidth',2);
h(3) = plot(xstar,mf_full,'k-','LineWidth',2);
h(4) = plot(X, y, 'rx');
title('After Hyperparameter Training');

figure;
imagesc(FullGP.K);

figure;
imagesc(GP.K);
