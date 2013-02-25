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
goal_function = SpaceTimeGoalFunction();
goal_function.expression = @(s,t)((cos(2*pi*t/2).^2).*(sin(s)./s+exp(s/10)));

%Limits for evaluation of the goal function
%Space (1D) from 0-12 and time from 0-4
goal_function.limits.low = [0.1 0];
goal_function.limits.high = [12 4];
goal_function.noiseStd = 0.1;
goal_function.noiseMean = 0;

%Get Ground Truth samples from the function to be plotted.
resolution.s = 0.2;
resolution.t = 0.05;
[GT_locations, t_s, t_t] = goal_function.GetEvaluationDomain(resolution);
GTy = goal_function.GetSample(GT_locations);

%% Plot the Ground Truth
figure;
surf(t_s,t_t,reshape(GTy,size(t_s)));

%% Training Data
N = 50; %Number of training Points.

X = goal_function.GetRandomEvaluationLocations(N);
y = goal_function.GetNoisySample(X);

%% Plot the Ground Truth with sample points
figure;
surf(t_s,t_t,reshape(GTy,size(t_s)));
hold on,
plot3(X.s,X.t,y,'o');

%% Set up Gaussian process

% Use a space-time GP regression model:
GP = tacopig.gp.STGP;

% Plug in the data
GP.X = X;
GP.y = y;

% Plug in the components
GP.MeanFn = tacopig.meanfn.STStationaryMean();
GP.CovFn   = tacopig.covfn.STSep(tacopig.covfn.SqExp,tacopig.covfn.ExpPeriodic(2));
GP.NoiseFn = tacopig.noisefn.Stationary();
GP.objective_function = @tacopig.objectivefn.NLML;

GP.solver_function = @anneal;

% Initialise the hyperparameters
GP.covpar   = 0.5*ones(1,GP.CovFn.npar(size(X,1)));
GP.meanpar  = zeros(1,GP.MeanFn.npar(size(X,1)));
GP.noisepar = 1e-1*ones(1,GP.NoiseFn.npar);


%% Before Learning: Query
GP.solve();

[mf, vf] = GP.query(GT_locations);
sf  = sqrt(vf);

%% Plot the mean and vf 
figure;
surf(t_s,t_t,reshape(mf,size(t_s)));
hold on
surf(t_s,t_t,reshape(mf+sf,size(t_s)),'facealpha',0.1);
surf(t_s,t_t,reshape(mf-sf,size(t_s)),'facealpha',0.1);
plot3(X.s,X.t,y,'o');

%% Learn & Query
%disp('Press any key to begin learning.')
%pause
GP.learn();
GP.solve();
[mf, vf] = GP.query(GT_locations);
sf  = sqrt(vf);

%% Plot the mean and vf 
figure;
surf(t_s,t_t,reshape(mf,size(t_s)));
hold on
surf(t_s,t_t,reshape(mf+sf,size(t_s)),'facealpha',0.1);
surf(t_s,t_t,reshape(mf-sf,size(t_s)),'facealpha',0.1);
plot3(X.s,X.t,y,'o');

% %% Generate samples from prior and posterior
% figure; subplot(1,2,1)
% xstar = linspace(-8,8,100);
% hold on;
% for i = 1:5
%     fstar = GP.sampleprior(xstar);
%     plot(xstar,fstar, 'color', rand(1,3));
% end
% title('Samples from Prior')
% 
% subplot(1,2,2)
% plot(X, y, 'k+', 'MarkerSize', 17)
% xstar = linspace(-8,8,100);
% hold on;
% for i = 1:5
%     fstar = GP.sampleposterior(xstar, 50+i);
%     plot(xstar,fstar, 'color', rand(1,3));
% end
% title('Samples from Posterior')

