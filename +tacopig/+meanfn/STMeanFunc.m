% Space-Time Mean Function Abstract Class
% All mean function classes must inherent from this class.

classdef STMeanFunc < tacopig.taco
    
    methods(Abstract)
        
        % Returns the number of parameters required by the mean function.
        % n = npar(D); 
        % Input :   D (dimensionality of the input data)
        % Output :  n (the number of parameters required by the mean function)
        n = npar(D);         


        % Returns the value of the mean function.
        % mu = eval(X, par); 
        % Input :  X1    - Struct with fields 's' and 't':
        %          X1.s  - D x N input points locations 1
        %          X1.t  - 1 x N input points timestamps 1
        %          par   - Mean function parameters.
        % Output : mu    - Mean function at the location of the input
        % points.
        mu = eval(X, par);


    end
    
    methods

        function gradient(this) 
        % Returns the mean gradient with respect to the parameters. Stub that may be overloaded
            error([class(this),' does not implement gradients!']);
        end
    end
    
    methods(Static, Hidden = true) 
        function theta = getMeanPar(GP) % Returns the mean parameter(s). 
            if isa(GP, 'tacopig.gp.GpCore')
                theta = GP.meanpar;
            elseif isa(GP, 'double') % Included to help with autotesting.
                theta = GP;
            else
                error('tacopig:badConfiguration', 'Error interpreting meanpar.');
            end
         end
        
    end
    
    
    
    
end    