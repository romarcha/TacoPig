% Stationary Mean Space Time GP Function
% The value of the mean is constant and learnt during training.

classdef STStationaryMean < tacopig.meanfn.STMeanFunc    
    
    methods(Static)
        function n = npar(~) 
            % Returns the number of parameters required by the class
            % n = GP.MeanFn.npar()
            % Always returns 1.
            n = 1; 
        end
        
        function mu = eval(X, GP) 
            % Returns the value of the mean at the location X
            %
            % mu = GP.MeanFn.eval(X, GP)
            %
            % Gp.MeanFn is an instantiation of the StationaryMean mean function class
            % Inputs: X1    - Struct with fields 's' and 't':
            %         X1.s  - D x N input points locations 1
            %         X1.t  - 1 x N input points timestamps 1
            % GP = The GP class instance can be passed to give the mean function access to its properties
            
            par = tacopig.meanfn.MeanFunc.getMeanPar(GP);
            if (numel(par)~=1)
                error('tacopig:inputInvalidLength','wrong number of hyperparameters!')
            end
            
            N = size(X.s,2);
            mu = par(1)*ones(1,N);
        end
        
        function g = gradient(X, ~) 
            %Evaluate the gradient of the mean function at locations X with respect to the parameters
            %
            % g = GP.MeanFn.gradient(X)
            %
            % Gp.MeanFn is an instantiation of the StationaryMean mean function class
            % Inputs:  X1    - Struct with fields 's' and 't':
            %          X1.s  - D x N input points locations 1
            %          X1.t  - 1 x N input points timestamps 1
            % Outputs: g = the gradient of the mean function at locations X with respect to the parameters (A cell of dimensionality 1 x Number of parameters. Each element is an array of dimensionality N x N)
            %
            % For this class g is a 1 x 1 cell array (because it only has 1 parameter) with the element being a 1 x N matrix.
            N = size(X.s,2);
            g = cell(1,1);
            g{1} = ones(1,N);
        end
    end
end    


