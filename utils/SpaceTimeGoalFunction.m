classdef SpaceTimeGoalFunction < handle
    % Goal Function
    % Models a generic goal function with space and time dependencies.
    
    properties
        expression
        noiseMean
        noiseStd
        limits
    end
    
    methods
        function y = GetSample(obj,X)
            
            y = feval(obj.expression,X.s,X.t);
        end
        
        function y = GetNoisySample(obj,X)
            noise = random('Normal',obj.noiseMean,obj.noiseStd,size(X.s));
            y = feval(obj.expression,X.s,X.t)+noise;
        end
        
        function X = GetRandomEvaluationLocations(obj,N)
            X.s = obj.limits.low(1) + (obj.limits.high(1)-...
                                  obj.limits.low(1)).*rand(1,N);
            X.t = obj.limits.low(2) + (obj.limits.high(2)-...
                                  obj.limits.low(2)).*rand(1,N);
        end
        
        function [GT, t_s, t_t] = GetEvaluationDomain(this,resolution)
            if(~isstruct(resolution))
                error('SpaceTimeGoalFunction: resolution is expected to be a struct with space and time components');
            end
            
            [t_s t_t] = meshgrid(this.limits.low(1):resolution.s:this.limits.high(1) ...
                                ,this.limits.low(2):resolution.t:this.limits.high(2));
            GT.s = t_s(:)';
            GT.t = t_t(:)';

        end
        
        
        

        
        function Plot(obj,numberOfSamples,color)
            if nargin < 3
                color = '--r';
            end
            domain = obj.GetEvaluationDomain(numberOfSamples);
            goalFunctionGTValues = obj.GetSample(domain);
            plot(domain,goalFunctionGTValues,color);
            title('Sample Goal Function');
        end
        
        function PlotWithRegression(obj,gp,mf,vf,samplePoints,sampleValues)
            f = [mf+2*sqrt(vf);flipdim(mf-2*sqrt(vf),1)];
            domain = obj.GetEvaluationDomain(length(mf));
            fill([domain'; flipdim(domain',1)], f, .7*[229 239 251]/255, 'EdgeColor', .7*[229 239 251]/255);
            hold on
            plot(domain,mf,'LineWidth',2);
            obj.Plot(100);
            hold on;
            plot(samplePoints,sampleValues,'rx','LineWidth',2);
            title(sprintf('Regression using GP with SE ARD params [%.3f %.3f] noise:%.3f',gp.covarianceParameters(1),gp.covarianceParameters(2),gp.noise));
        end
    end
end

