% The Neural Network covariance function class
%
% Example Instantiation:
% GP.CovFn = tacopig.covfn.NeuralNetwork();
% Creates an instance of the neural network covariance function as the CovFn 
% property of an instantiation of a Gaussian process class named GP.
%
% Note on hyperparameters:
% It has D+1 hyperparameter, one for each dimension and one for the signal
% variance. It has the form [Lengthscales, Sigma_f]

classdef NeuralNetwork < tacopig.covfn.CovFunc
    
    properties(Constant)
        ExampleUsage = 'tacopig.covfn.NeuralNetwork()'; %Instance of class created for testing
    end
    
    
    methods 
        
        function this = NeuralNetwork() 
            % Neural Network covariance function constructor
            % GP.CovFn = tacopig.covfn.NeuralNetwork()
            % Gp.CovFn is an instantiation of the NeuralNetwork covariance function class
        end   
        
        function n_theta = npar(this,D)
            % Returns the number of hyperparameters required by all of the children covariance functions
            %
            % n_theta = GP.covfunc.npar(D)
            %
            % Gp.CovFn is an instantiation of the SqExp covariance function class
            % Inputs: D (Dimensionality of the dataset)
            % Outputs: n_theta (number of hyperparameters required by all of the children covariance function)
            
            if D <= 0
              error('tacopig:inputOutOfRange', 'Dimension cannot be < 1');
            end
            if mod(D,1) ~= 0
              error('tacopig:inputInvalidType', 'Dimension must be an integer');
            end
            n_theta = D+1; 
        end
        
        function K = eval(this, X1, X2, GP)
            % Get covariance matrix between input sets X1,X2 
            %
            % K = GP.covfunc.eval(X1, X2, GP)
            %
            % Gp.CovFn is an instantiation of the SqExp covariance function class
            % Inputs:  X1 = D x N Input locations
            %          X2 = D x M Input locations
            %          GP = The GP class instance can be passed to give the covariance function access to its properties
            % Outputs: K = covariance matrix between input sets X1,X2 (N x M)

            par = this.getCovPar(GP);
            [D,N1] = size(X1); %dimensionality and number of points in X1
            N2 = size(X2,2); %number of points in X2
            if D~=size(X2,1)
                error('tacopig:dimMismatch','Dimensionality of X1 and X2 must be the same');
            end
            if (length(par)~=2)
                error('tacopig:inputInvalidLength','Wrong number of hyperparameters for Neural Network');
            end
            
            w = par(1:D)'.^(-2);
            
            X1X2 = 2*X1'*diag(w)*X2;
            XX1 = sum(X1.*w(:,ones(1,N1)).*X1,1);
            XX2 = sum(X2.*w(:,ones(1,N2)).*X2,1);
            
            z = X1X2./sqrt((1+2*XX1)'*(1+2*XX2));

            K = (par(end).^2)*asin(z);
        end
        
        function [g] = gradient(this,X, GP)
         % Returns gradient of the covariance matrix k(X,X) with respect to each hyperparameter
        % g = Gp.CovFn.gradient(X, GP)
        %
        % Gp.CovFn is an instantiation of the SqExp covariance function class
        %
        % Inputs:   X = Input locations (D x N matrix)
        %           GP = The GP class instance can be passed to give the covariance function access to its properties
        % Outputs:  g = gradient of the covariance matrix k(X,X) with respect to each hyperparameter. Cell Array (1 x Number of Hyperparameters). Each Element is a N x N matrix

            par = this.getCovPar(GP);
            % Same as K?
            Kg = this.eval(X, X, par);
            
            [d,n] = size(X);
            g = cell(1,d+1);
            for i=1:d
                %Compute weighted squared distance
                row = X(i,:);
                XX = row.*row;
                XTX = row'*row;
                XX = XX(ones(1,n),:);
                z = max(0,XX+XX'-2*XTX);
                w = par(i)^(-3);
                g{i} = w*Kg.*z;
            end
            g{d+1} = Kg*(2/par(end));
        end
        
        
        % Also overload the point covariance kx*x* - its trivial
        function v = pointval(this, x_star, GP)
            % Efficient case for getting diag(k(X,X))
            % v = Gp.CovFn.pointval(X, GP)
            %
            % Gp.CovFn is an instantiation of the SqExp covariance function class
            %
            % Inputs : X = Input locations (D x n matrix)
            % Output : v = diag(k(X,X))
            
            par = this.getCovPar(GP);
            [D,N1] = size(x_star); %number of points in X1
            if (length(par)~=D+1)
               error('tacopig:inputInvalidLength','Wrong number of hyperparameters!');
            end
            v = par(end).^2 * ones(1,size(x_star,2));
        end
        
    end
end
