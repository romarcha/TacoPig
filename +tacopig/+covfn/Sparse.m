% Sparcifies a covariance function.
% By making use of [1] the original covariance matrix is multiplied by the
% factor max{0,(1-norm(x-x')/theta)^v} where theta and v are parameters
% that determine the cutoff point and assure positive semidefinitiveness.
%
% [1] Hamers, Suykens - 2002 - Compactly supported RBF Kernels for
% Sparsifying the Gram Matrix
%
% Usage: tacopig.covfn.Sparse(covfn,theta,v);
% 
% Example instantiation:
% GP.CovFn   = tacopig.covfn.Sparse(tacopig.covfn.SqExp(),1,2);
% Instantiates a sparse version of the Squared Exponential covariance
% function.


classdef Sparse < tacopig.covfn.CovFunc
    
    properties(Constant)
        %Instance of class created for testing
        ExampleUsage = 'tacopig.covfn.Sparse(tacopig.covfn.SqExp(), 3, 1)';
    end
    
    properties
       theta    % cutoff parameter see [1]
       v        % dimension exponent see [1]
       covfn    % the constructor of the covariance function that is 
                % sparcified
    end
    
    methods
        
        function this = Sparse(covfn, theta, v)
           % Sparse covariance function constructor
           %
           % GP.CovFn = tacopig.covfn.Sparse(covfn, theta, v)
           %
           % Inputs:   covfn     the constructor of the covariance function
           %                     to be sparcified
           %           theta      index of hypermeter(s) to be clamped
           %           v     value(s) of clamped hypermeter(s)
       
           if ~isa(covfn,'tacopig.covfn.CovFunc')
               error('tacopig:badConfiguration', [class(this),...
                   ': must specify a valid covariance function']); 
           end
           this.theta = theta;
           this.v = v;
           this.covfn = covfn;
        end   
            
        function n_theta = npar(this,D)
            % Returns the number of free hyperparameters required by the covariance function
            %
            % n_theta = GP.CovFn.npar(D)
            %
            % GP.CovFn is an instantiation of the Clamp covariance function class
            % Inputs: D = Dimensionality of the input data
            % Outputs: n_theta = the number of free hyperparameters required by the covariance function
            
            n_theta = this.covfn.npar(D);
        end
        
        function K = eval(this, X1, X2, GP)
            %Get covariance matrix between input sets X1,X2
            %
            % K = GP.CovFn.eval(X1, X2, GP)
            %
            % GP.CovFn is an instantiation of the Clamp covariance function class
            % Inputs:  X1 = Input locations (D x N matrix)
            %          X2 = Input locations (D x M matrix)
            %          GP = The GP class instance can be passed to give the covariance function access to its properties
            % Outputs: K = covariance matrix between input sets X1,X2 (N x M)
            
            par = this.getCovPar(GP);
            if (length(par)~=this.npar(size(X1,1)))
                error('tacopig:inputInvalidLength','Wrong number of hyperparameters!');
            end
            [D,N1] = size(X1); %number of points in X1
            N2 = size(X2,2); %number of points in X2
            if (D~=size(X2,1))
                error('tacopig:dimMismatch','Dimensionality of X1 and X2 must be the same.');
            end
            
            K = this.covfn.eval(X1,X2,par);
            
            %Compute weighted squared distances:
            w = (this.theta(1:D).*par(1:D))'.^-1;
            XX1 = sum(w(:,ones(1,N1)).*X1.*X1,1);
            XX2 = sum(w(:,ones(1,N2)).*X2.*X2,1);
            X1X2 = (w(:,ones(1,N1)).*X1)'*X2;
            XX1T = XX1';
            % numerical effects can drive z slightly negative 
            z = sqrt(max(0,XX1T(:,ones(1,N2)) + XX2(ones(1,N1),:) - 2*X1X2));
            z = max(0,(1-z).^this.v);
            K = K.*sparse(z);
        end
        
        function K = Keval(this, X, GP)
            % Evaluation of k(X,X) (symmetric case)
            % Get covariance matrix between input sets X1,X2
            %
            % K = GP.CovFn.Keval(X, GP)
            %
            % GP.CovFn is an instantiation of the Clamp covariance function class
            %
            % Inputs:  X = Input locations (D x N matrix)
            %          GP = The GP class instance can be passed to give the covariance function access to its properties
            % Outputs: K = covariance matrix between input sets X and itself (N x N)   
            
            par = this.getCovPar(GP);
            
            if (length(par)~=this.npar(size(X,1)))
                error('tacopig:inputInvalidLength','Wrong number of hyperparameters!');
            end

            K = this.eval(X,X,par);
        end
        
        function g = gradient(this,X, GP)
            % Returns gradient of the covariance matrix k(X,X) with respect to each hyperparameter
            % g = GP.CovFn.gradient(X, GP)
            %
            % GP.CovFn is an instantiation of the Clamp covariance function class
            %
            % Inputs:   X = Input locations (D x N matrix)
            %           GP = The GP class instance can be passed to give the covariance function access to its properties
            % Outputs:  g = gradient of the covariance matrix k(X,X) with respect to each hyperparameter. Cell Array (1 x Number of Hyperparameters). Each Element is a N x N matrix


            
            par = this.getCovPar(GP);
            % Inefficiency at the cost of being clean
            % The alternative is to have a gradient method that takes as an
            % input a list of which gradients we need to know...
            
            error('Sparse Covariance Matrix gradient not implemented');
            
        end
        
        % Overload the point covariance - its trivial to add them
        function v = pointval(this, x_star, GP)
            % Efficient case for getting diag(k(X,X))
            % v = GP.CovFn.pointval(X, GP)
            %
            % GP.CovFn is an instantiation of the Clamp covariance function class
            %
            % Inputs : X = Input locations (D x n matrix)
            %          GP = The GP class instance can be passed to give the covariance function access to its properties
            % Output : v = diag(k(X,X))
            par = this.getCovPar(GP);
            if (length(par)~=this.npar(size(x_star,1)))
                error('tacopig:inputInvalidLength','Wrong number of hyperparameters!');
            end
            v = this.covfn.pointval(x_star, par);
        end
    end
end

