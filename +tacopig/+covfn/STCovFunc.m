% Space-Time Covariance Function Abstract Class
% All space-time covariance function classes must inherent from this class.

classdef STCovFunc < tacopig.taco
    
    methods (Abstract)
        % Automatically report hyperparameter dimensionality
        % 
        % n_theta = GP.covfn.npar(D); 
        %
        % Gp.CovFn is an instantiation of the covariance function class
        % Inputs: D = dimensionality of spatial component int the dataset
        % Outputs: n_theta = number of hyperparameters required (including
        % time)
        n_theta = npar(this, D); 
        
        % Get covariance matrix between input sets X1,X2
        %
        % K = Gp.CovFn.eval(X1, X2, theta);
        %
        % Gp.CovFn is an instantiation of the covariance function class 
        % X1    - Struct with fields 's' and 't':
        % X1.s  - D x N input points locations 1
        % X1.t  - 1 x N input points timestamps 1
        % X2    - Struct with fields 's' and 't':
        % X2.s  - D x M input points locations 2
        % X2.t  - 1 x M input points timestamps 2
        % theta - 1 x L vector of hyperparameters (L is npar for this cov func)
        K = eval(this, X1, X2, theta);
    end

    methods

        % just run with the default constructor   
        
        
        function K = Keval(this, X, GP)
        % Evaluation of k(X,X) (symmetric case)
        %
        % K = GP.covfn.Keval(X, GP)
        % 
        % Inputs :  X1    = Struct with fields 's' and 't':
        %           X1.s  = D x N input points locations
        %           X1.t  = 1 x N input points timestamps
        %           GP    = The GP class instance can be passed to give the covariance function access to its properties
        % Outputs : K     = covariance matrix between input sets X and itself (N x N)   
            K = this.eval(X,X,GP);
        end
        
        
        function gradient(this)
            % Returns gradient of k(X,X) with respect to each hyperparameter
            error('tacopig:badConfiguration',[class(this),' does not implement gradients!']);
        end
        
        function theta = getCovPar(this, GP)
            % Returns the covariance function's hyperparameters
            if isa(GP, 'tacopig.gp.GpCore')
                theta = GP.covpar;
            elseif isa(GP, 'double')
                theta = GP;
            else
                error('tacopig:badConfiguration', 'Error interpreting covpar.');
            end
        end
        
        function CheckSpaceTimeInput(this,X)
            if(isa(X,'struct'))
                if(~isfield(X,'s'))
                    error('STSep::eval X does not have space field (s)');
                end
                if(~isfield(X,'t'))
                    error('STSep::eval X does not have time field (t)');
                end
            else
                error('STSep::eval X is not space-time data');
            end
        end
        
    end
end    
