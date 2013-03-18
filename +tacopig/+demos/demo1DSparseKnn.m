%%
% Gaussian Process Demo Script for Knn
%
% In this demo, a kdtree structure is used to 
% organize a large amount of datapoints. Assuming that 'near' datapoints
% are correlated strongly, for each prediction point, the covariance matrix
% is only built with a subset of points, using the k-nearest-neighbors
% given by the kd-tree structure.
%
% To do: include the kdtree structure inside a special kind of GP called 
%

%Add optimization folder
if ~exist('minfunc')
    fprintf('It looks like you need to add minfunc to your default path...\n');
    fprintf('(Add tacopig/optimization/minfunc{/mex} to pathdef.m for permanent access)\n');
    fprintf('Press any key to attempt to continue...\n');
    pause();
end
if ~exist('flann_build_index')
    error('It looks like you do not have flann installed...\n');
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
goalFunction.noiseStd = 0.4;

%%
% Plot of the goal function values
%figure;
%goalFunction.Plot(1000);

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
N = 110000;

%%
% Location of sample points
samplePoints = goalFunction.GetRandomEvaluationLocations(N);

%%
% Get Noisy samples
sampleValues = goalFunction.GetNoisySample(samplePoints);

%%
% % Plot original ground truth and sample values with noise
% figure;
% goalFunction.Plot(numberOfSamples)
% hold on;
% plot(samplePoints,sampleValues,'rx');
% title(sprintf('Sample values with noise mean %.3f and std %.3f',goalFunction.noiseMean,goalFunction.noiseStd));

xstar = goalFunction.GetEvaluationDomain(200);

% Order data points for visualisation
X = samplePoints';
y = sampleValues';
%[X id] = sort(X);
%y = y(id);
% X(5000:8000) = [];
% y(5000:8000) = [];
% X(500:600) = [];
% y(500:600) = [];

%% Full Gaussian process
% Use a standard GP regression model:
clear GP;
GP = tacopig.gp.Regressor;
GP.MeanFn  = tacopig.meanfn.FixedMean(mean(y));
GP.CovFn   = tacopig.covfn.SqExp();
GP.NoiseFn = tacopig.noisefn.Stationary();
GP.objective_function = @tacopig.objectivefn.NLML;
GP.opts.numDiff = 1; %Derivatives are calculated numerically

%Initial HyperParameters
GP.covpar   = [0.0713,0.5575];
GP.meanpar  = zeros(1,GP.MeanFn.npar(size(X,1)));
GP.noisepar = 0.0983;

% Learn hyperparameters from random 700 points
n_train_hyper = 700;
if(length(X) > n_train_hyper)
    indexes = randsample(length(X),n_train_hyper);
    GP.X = X(:,indexes);
    GP.y = y(indexes);
else
    GP.X = X;
    GP.y = y;
end
GP.solve();
GP.learn();

%% Query
knn = 400;
m_f = zeros(1,length(xstar));
v_f = zeros(1,length(xstar));
s_f = zeros(1,length(xstar));
s_f_noise = zeros(1,length(xstar));
knn_params.algorithm = 'linear';
flann_set_distance_type(1);
[index, parameters] = flann_build_index(X, knn_params);
for i = 1:length(xstar)
    [result, dists] = flann_search(index,xstar(i),knn,parameters);
    
    % Plug in the data
    GP.X = X(result);
    GP.y = y(result);
    GP.solve();
    [m_f(i), v_f(i)] = GP.query(xstar(i));
    s_f(i)  = sqrt(v_f(i));
    s_f_noise(i)  = sqrt(v_f(i)+GP.noisepar^2);
end

%% Display learnt model
figure;
f  = [m_f+2*(s_f),flipdim(m_f-2*(s_f),2)]';
h(1) = fill([xstar, flipdim(xstar,2)], f, [6 6 6]/8, 'EdgeColor', [6 6 6]/8);
hold on
h(2) = plot(xstar,m_f,'k-','LineWidth',2);
h(2) = plot(xstar,m_f+2*(s_f_noise),'k:','LineWidth',2);
h(2) = plot(xstar,m_f-2*(s_f_noise),'k:','LineWidth',2);
if length(X) > 600
    h(3) = plot(X, y, 'k.');
else
    h(3) = plot(X, y, 'ko');
end
goalFunction.Plot(1000);

