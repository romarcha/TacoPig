% A special case of the Matern covariance function class with \nu set to 3/2 
% and considering inputs to be geographic coordinates of the form Lat,
% Long.
% i.e. to avoid calculating distance distortions it calculates the distance
% based on geodecic approximation provided by an external library.
%
% Important: Only works with 2 dimentional input data! (Lat,Lon) and has
% one hyperparameter of lengthscale and one for signal variance.
%
% Example instantiation:
% GP.CovFn   = tacopig.covfn.Mat3Geo();
% Instantiates a Matern3 covariance function (as a property of a Gaussian process instantiation called GP)
%
%
% k(X1,X2) = Sigma_f^2*(1+sqrt(3)*r/l))*exp(-sqrt(3)*r/l);
%
% where r = diatanceOverTheEart(X1,X2); and each X1 has its own latitude
% and longitude.
% i.e. X1 and X2 are input matrices of dimensionality 2 x N and 2 x M, respectively. 
% N and M are the number of observations in the matrix X1 and X2, respectively.
% k(X1,X2) is a N x M covariance matrix
%
% Note on hyperparameters:
% The hyperparameter vector has a dimensionality of 2 and is of the form [Lengthscale, Sigma_f]
% Sigma_f is the signal variance hyperpameter 

classdef Mat3Geo < tacopig.covfn.CovFunc
    
    properties(Constant)
        ExampleUsage = 'tacopig.covfn.Mat3Geo()'; %Instance of class created for testing
    end
    
   
    methods
        
        
        function this = Mat3Geo() 
            % Matern3 covariance function constructor
            % GP.CovFn = tacopig.covfn.Mat3Geo()
            % Gp.CovFn is an instantiation of the Mat3Geo covariance function class
            
            % First check if function to calculate distance over the earth
            % is present or not.
            if ~exist('geodesicinverse')
                fprintf('It looks like you need to add geodesic library of matlab...\n');
            end
        end 
        
        function n_theta = npar(this, D)
            % Returns the number of hyperparameters required by all of the children covariance functions
            %
            % n_theta = GP.covfunc.npar(D)
            %
            % Gp.CovFn is an instantiation of the Mat3 covariance function class
            % Inputs: D (Dimensionality of the dataset)
            % Outputs: n_theta (number of hyperparameters required by all of the children covariance function)

            if D <= 0
              error('tacopig:inputOutOfRange', 'Dimension cannot be < 1');
            end
            if mod(D,1) ~= 0
              error('tacopig:inputInvalidType', 'Dimension must be an integer');
            end
            n_theta = 2; % one lengthscale + signal variance
        end
        
        function K = eval(this, X1, X2, GP)
            % Get covariance matrix between input sets X1,X2 
            %
            % K = GP.covfunc.eval(X1, X2, GP)
            %
            % Gp.CovFn is an instantiation of the Mat3Geo covariance function class
            % Inputs:  X1 = 2 x N Input locations
            %          X2 = 2 x M Input locations
            %          GP = The GP class instance can be passed to give the covariance function access to its properties
            % Outputs: K = covariance matrix between input sets X1,X2 (N x M)            
        
            par = this.getCovPar(GP);
            [D1,N1] = size(X1); %dimensionality and number of points in X1
            [D2,N2] = size(X2); %number of points in X2
            if D1~=2 || D2~=2
                error('tacopig:dimMismatch','Dimensionality of X1 and X2 must be 2');
            end
            if (length(par)~=2)
                error('tacopig:inputInvalidLength','Wrong number of hyperparameters for Mat3Geo');
            end
            %Compute geographic distance in kilometers:
            dists = zeros(N1,N2);
            for i = 1:N1
                current = X1(:,i)';
                input = [repmat(current,N2,1) X2'];
                answer = geodesicinverse(input);
                dists(i,:) = answer(:,3)'*10^-3; %%*10^-3 transforms to km
            end
            l = par(1);
            K = par(2)^2 *(1+sqrt(3)*dists/l).* exp(-sqrt(3)*dists/l); 
        end
        
        function [g] = gradient(this, X, GP)
            % Returns gradient of the covariance matrix k(X,X) with respect to each hyperparameter
            % g = Gp.CovFn.gradient(X, GP)
            %
            % Gp.CovFn is an instantiation of the Mat3 covariance function class
            %
            % Inputs:   X = Input locations (D x N matrix)
            %           GP = The GP class instance can be passed to give the covariance function access to its properties
            % Outputs:  g = gradient of the covariance matrix k(X,X) with respect to each hyperparameter. Cell Array (1 x Number of Hyperparameters). Each Element is a N x N matrix
            
            error('Gradient not implemented.');
        end
        
        
        % Also overload the point covariance kx*x* - its trivial
        function v = pointval(this, x_star, GP)
            % Efficient case for getting diag(k(X,X))
            % v = Gp.CovFn.pointval(X, GP)
            %
            % Gp.CovFn is an instantiation of the Mat3 covariance function class
            %
            % Inputs : X = Input locations (D x n matrix)
            % Output : v = diag(k(X,X))
            
             par = this.getCovPar(GP);
             [D,N1] = size(x_star); %number of points in X1
             if (length(par)~=2)
                error('tacopig:inputInvalidLength','Wrong number of hyperparameters!');
             end
             if (D~=2)
                error('tacopig:inputInvalidLength','Wrong number of dimentions on xstar, should be two for geographic covariance function!');
             end
            v = par(end).^2 * ones(1,size(x_star,2));
        end
        
    end
end
