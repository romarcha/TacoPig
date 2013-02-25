%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                   Gaussian Process Demo Script                        %
%  Demonstrates GP regression using the taco-pig toolbox on 1-D         %
%                      spatial temporal data.                           %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Add optimization folder
if ~exist('minfunc')
    fprintf('It looks like you need to add minfunc to your default path...\n');
    fprintf('(Add tacopig/optimization/minfunc{/mex} to pathdef.m for permanent access)\n');
    fprintf('Press any key to attempt to continue...\n');
    pause();
end
p = pwd(); slash = p(1);
addpath(genpath(['..',slash,'optimization']))

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%% 1-D Example%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
close all; clc;

%% Set up 1-D Data

%% Ground Truth

%Ideal function where to take samples from
goal_function = GoalFunction();
goal_function.expression = @(x)(cos(2*pi*x/2));

%Limits for evaluation of the goal function
%Space (1D) from 0-12 and time from 0-4
goal_function.limits.low = 0;
goal_function.limits.high = 4;
goal_function.noiseStd = 0.1;
goal_function.noiseMean = 0;

%Get Ground Truth samples from the function to be plotted.
numberOfSamples = 500;
GT_locations = goal_function.GetEvaluationDomain(numberOfSamples);
GTy = goal_function.GetSample(GT_locations);

%% Plot the Ground Truth
figure;
plot(GT_locations,GTy);

%% Training Data
N = 30; %Number of training Points.

X = goal_function.GetRandomEvaluationLocations(N);
y = goal_function.GetNoisySample(X);

%% Plot the Ground Truth with sample points
figure;
plot(GT_locations,GTy);
hold on,
plot(X,y,'o');

%% Set up Gaussian process

% Use a space-time GP regression model:
GP = tacopig.gp.Regressor;

% Plug in the data
X = X';
y = y';
GP.X = X;
GP.y = y;

% Plug in the components
GP.MeanFn = tacopig.meanfn.StationaryMean();
GP.CovFn   = tacopig.covfn.Sum(tacopig.covfn.ExpPeriodic(2),tacopig.covfn.SqExp());
GP.NoiseFn = tacopig.noisefn.Stationary();
GP.objective_function = @tacopig.objectivefn.NLML;

GP.solver_function = @anneal;

% Initialise the hyperparameters
GP.covpar   = [1 1 0.5 0.5];%;0.5*ones(1,GP.CovFn.npar(size(X,1)));
GP.meanpar  = zeros(1,GP.MeanFn.npar(size(X,1)));
GP.noisepar = 1e-1*ones(1,GP.NoiseFn.npar);

%% Query
GP.solve();
[mf_p, vf_p] = GP.query(GT_locations);
sf_p  = sqrt(vf_p);

%% Learn & Query
GP.learn();
GP.solve();
[mf, vf] = GP.query(GT_locations);
sf  = sqrt(vf);

%% Plot the mean and vf 
figure;
%Learnt model
f = [mf+2*sf;flipdim(mf-2*sf,1)];
fill([GT_locations; flipdim(GT_locations,1)], f, [0 0 255]/255, 'EdgeColor', [0 0 255]/255,'edgealpha',0.1);
hold on
plot(GT_locations,mf,'LineWidth',2);
f_p = [mf_p+2*sf_p;flipdim(mf_p-2*sf_p,1)];
h = fill([GT_locations; flipdim(GT_locations,1)], f_p, [255 0 0]/255,'FaceColor',[255 0 0]/255, 'EdgeColor', [255 0 0]/255,'edgealpha',0.1);
hold on
plot(GT_locations,mf_p,'r','LineWidth',2);
plot(GT_locations,GTy,'k','LineWidth',2);
hold on;
plot(X,y,'rx','LineWidth',2);
title('After Learning');
