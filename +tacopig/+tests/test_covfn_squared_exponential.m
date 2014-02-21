data_41 = [ -2.1775;
            -0.9235;
             0.7502;
            -5.8868];
        
data_51 = [-2.7995;
            4.2504;
            2.4582;
            6.1426;
           -4.0911];

data_42 = [- 2.142,  1.543;
             5.186, -1.243;
             1.475, -3.485;
           -12.348,  2.148];
       
data_62 = [3.214, -1.025;
           5.126,  5.147;
          -1.025, -4.123;
           1.257, -5.147;
           2.354, -2.746;
           1.288,  3.146];

%% Test 1Dim

%%
% Set up Gaussian process

% Use a standard GP regression model:
GP = tacopig.gp.Regressor;

% Plug in the data
GP.X = data_41';
GP.y = zeros(size(data_41));

% Plug in the components
GP.MeanFn  = tacopig.meanfn.FixedMean(mean(GP.y));
GP.CovFn   = tacopig.covfn.SqExp();
GP.NoiseFn = tacopig.noisefn.Stationary();
GP.objective_function = @tacopig.objectivefn.NLML;

% Initialise the hyperparameters
GP.covpar   = [0.5 1.5];
GP.meanpar  = zeros(1,GP.MeanFn.npar(size(GP.X,1)));
GP.noisepar = 1e-1*ones(1,GP.NoiseFn.npar);

K_1 = GP.CovFn.eval(GP.X,GP.X,GP)

%% Test 2Dim

% Plug in the data
GP.X = data_42';
GP.y = zeros(size(data_42));

GP.covpar   = [0.5 0.75 1.5];

K_2 = GP.CovFn.eval(data_42',data_62',GP)

for i = 1:size(data_42,1)
    for j = 1:size(data_62,1)
        K_2_element(i,j) = GP.CovFn.eval(data_42(i,:)',data_62(j,:)',GP);
        
        x1 = data_42(i,:)';
        x2 = data_62(j,:)';
        diff = (x1-x2);
        lengthscale_matrix = diag(GP.covpar(1:2).^(-2));
        wsd = diff'*lengthscale_matrix*diff;
        sigma_f = GP.covpar(3);
        K_2_true(i,j) = (sigma_f^2)*exp(-0.5*wsd);
    end
end

K_2_element
K_2_true