% Defines a separable, space-time covariance function.
% tacopig.covfn.STSep(spaceCovFunc, timeCovFunc)
%
% spaceCovFunc and timeCovFunc must inherit from tacopig.covfn.CovFn
%
% Example instantiation:
% GP.CovFn   = tacopig.covfn.STSep(tacopig.covfn.SqExp(),tacopig.covfn.Mat3());
% Instantiates a separable covariance function, where the squared
% exponential covariance function is used for space and the Matern3 cov
% function is used for time.

classdef STSep < tacopig.covfn.STCovFunc
    
    properties(Constant)
        ExampleUsage = 'tacopig.covfn.STSep(tacopig.covfn.SqExp(),tacopig.covfn.Mat3());'; %Instance of class created for testing
    end
    
    properties
       sCovFunc % The spatial covariance function.
       tCovFunc % The temporal covariance function
    end
    
    methods
        
        function this = STSep(spaceCovFunc, timeCovFunc)
            % STSep covariance function constructor
            % GP.CovFn = tacopig.covfn.STSep(spaceCovFunc, timeCovFunc)
            % Inputs: spaceCovFunc = space covariance function
            %         timeCovFunc  = time covariance function
            if ~isa(spaceCovFunc,'tacopig.covfn.CovFunc')
                error('Space CovFunc is not a valid covariance function');
            end
            if ~isa(timeCovFunc,'tacopig.covfn.CovFunc')
                error('Time CovFunc is not a valid covariance function');
            end
            this.sCovFunc = spaceCovFunc;
            this.tCovFunc = timeCovFunc;
        end   
            
        function n_theta = npar(this,D)
            % Returns the number of hyperparameters required by the space
            % and time covariance functions
            %
            % n_theta = GP.covfunc.npar(D)
            %
            % Gp.CovFn is an instantiation of the Sum covariance function class
            % Inputs: D (Dimensionality of the space dimentions dataset)
            % Outputs: n_theta (number of hyperparameters required by all of the children covariance function)
            
            %Number of parameters for the spatial covariance function.
            n_theta = this.sCovFunc.npar(D);
            
            %Number of parameters for the temporal covariance function
            %(only has one dimention that is time)
            n_theta = n_theta + this.tCovFunc.npar(1);
        end
        
        function K = eval(this, X1, X2, GP)
            %Get covariance matrix between input sets X1,X2 
            %
            % K = GP.covfunc.eval(X1, X2, GP)
            %
            % Gp.CovFn is an instantiation of the Sum covariance function class
            % Inputs:   X1    = Struct with fields 's' and 't':
            %           X1.s  = D x N input points locations 1
            %           X1.t  = 1 x N input points timestamps 1
            %           X2    = Struct with fields 's' and 't':
            %           X2.s  = D x M input points locations 2
            %           X2.t  = 1 x M input points timestamps 2
            % theta - 1 x L vector of hyperparameters (L is npar for this cov func)
            %          GP = The GP class instance can be passed to give the covariance function access to its properties
            % Outputs: K = covariance matrix between input sets X1,X2 (N x M)

            par = this.getCovPar(GP);
            
            % Check that X1 and X2 have the required properties:
            this.CheckSpaceTimeInput(X1);
            this.CheckSpaceTimeInput(X2);

            D = size(X1.s,1); %spatial dimensionaly of points in X1
            
            if D~=size(X2.s,1)
                 error('tacopig:dimMismatch','Space dimensionality of X1 and X2 must be the same.');
            end
            
            
            npar = length(par);
            if (npar~=this.npar(D))
                error('tacopig:inputInvalidLength','Wrong number of hyperparameters for STSep');
            end
            
            %Calculate the spatial component of the covariance matrix
            K = this.sCovFunc.eval(X1.s,X2.s, par(1:this.sCovFunc.npar(D)));
            left = this.sCovFunc.npar(D)+1;
            
            %Calculate the temporal component and multiply it with the 
            %spatial one.
            K = K.*this.tCovFunc.eval(X1.t,X2.t, par(left:end));
            
        end
        
        
        function K = Keval(this, X, GP)
            % Evaluation of k(X,X) (symmetric case)
            % 
            % K = Keval(X, GP)
            %
            % Gp.CovFn is an instantiation of the STCovFunc covariance function class
            %
            % Inputs:   X1    = Struct with fields 's' and 't':
            %           X1.s  = D x N input points locations 1
            %           X1.t  = 1 x N input points timestamps 1
            %           X2    = Struct with fields 's' and 't':
            %           X2.s  = D x M input points locations 2
            %           X2.t  = 1 x M input points timestamps 2
            %           GP = The GP class instance can be passed to give the covariance function access to its properties
            % Outputs:  K = covariance matrix between input sets X and itself (N x N)

            par = this.getCovPar(GP);
            
            % Check that X is proper structure
            this.CheckSpaceTimeInput(X);
            
            D = size(X.s,1); %dimension of X
            
            npar = length(par);
            if (npar~=this.npar(D))
                error('tacopig:inputInvalidLength','Wrong number of hyperparameters for NegExp');
            end

            %Calculate the spatial component of the covariance matrix
            K = this.sCovFunc.Keval(X.s, par(1:this.sCovFunc.npar(D)));
            left = this.sCovFunc.npar(D)+1;
            %Calculate the temporal component and multiply it with the 
            %spatial one.
            K = K.*this.tCovFunc.Keval(X.t, par(left:end));
        end
        
        % Overload the point covariance - its trivial to add them
        function v = pointval(this, x_star, GP)
            % Efficient case for getting diag(k(X,X))
            % v = Gp.CovFn.pointval(X, GP)
            %
            % Gp.CovFn is an instantiation of the Sum covariance function class
            %
            % Inputs : X = Input locations (D x n matrix)
            % Output : v = diag(k(X,X))
            
            par = this.getCovPar(GP);
            
            [D] = size(x_star.s,1);
            
            npar = length(par);
            if (npar~=this.npar(D))
                error('tacopig:inputInvalidLength','Wrong number of hyperparameters for NegExp');
            end
            
            v = this.sCovFunc.pointval(x_star.s,par(1:this.sCovFunc.npar(D)));
            left = this.sCovFunc.npar(D)+1;
            v = v.*this.tCovFunc.pointval(x_star.t,par(left:end));
            
        end
        
        function g = gradient(this,X, GP)
         % Returns gradient of the covariance matrix k(X,X) with respect to each hyperparameter
        % g = Gp.CovFn.gradient(X, GP)
        %
        % Gp.CovFn is an instantiation of the Sum covariance function class
        %
        % Inputs:   X = Input locations (D x N matrix)
        %           GP = The GP class instance can be passed to give the covariance function access to its properties
        % Outputs:  g = gradient of the covariance matrix k(X,X) with respect to each hyperparameter. Cell Array (1 x Number of Hyperparameters). Each Element is a N x N matrix

            par = this.getCovPar(GP);
            [D,~] = size(X.s);
            glist = cell(2,1);
            glist{1} = this.sCovFunc.gradient(X.s, par(1:this.sCovFunc.npar(D)) );
            left = this.sCovFunc.npar(D)+1;
            glist{2} = this.tCovFunc.gradient(X.t, par(left:end) );
            
            g = cat(2,glist{:}); % be careful to concatenate along the right dimension...
            
        end
        
    end
end
