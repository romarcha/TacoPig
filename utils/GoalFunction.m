classdef GoalFunction < handle
    %GOALFUNCTION Goal Function to be optimized
    %   
    
    properties
        expression
        noiseMean
        noiseStd
        limits
    end
    
    methods
        function y = GetSample(obj,x)
            y = feval(obj.expression,x);
        end
        function y = GetNoisySample(obj,x)
            noise = random('Normal',obj.noiseMean,obj.noiseStd,size(x));
            y = feval(obj.expression,x)+noise;
        end
        function X = GetRandomEvaluationLocations(obj,N)
            X = obj.limits.low + (obj.limits.high-...
                                  obj.limits.low).*rand(N,1);
        end
        function domain = GetEvaluationDomain(obj,numberOfSamples)
            domain = linspace(obj.limits.low,obj.limits.high,numberOfSamples);
        end
        function Plot(obj,numberOfSamples)
            domain = obj.GetEvaluationDomain(numberOfSamples);
            goalFunctionGTValues = obj.GetSample(domain);
            plot(domain,goalFunctionGTValues,'--r');
            title('Sample Goal Function');
        end
        function PlotWithRegression(obj,gp,mf,vf,samplePoints,sampleValues)
            f = double([mf+2*sqrt(vf);flipdim(mf-2*sqrt(vf),1)]);
            domain = obj.GetEvaluationDomain(100);
            fill([domain'; flipdim(domain',1)], f, [229 239 251]/255, 'EdgeColor', [229 239 251]/255);
            hold on
            plot(domain,double(mf));
            obj.Plot(100);
            hold on;
            plot(double(samplePoints),double(sampleValues),'rx');
            title(sprintf('Regression using GP with SE ARD params [%.3f %.3f] noise:%.3f',gp.covarianceParameters(1),gp.covarianceParameters(2),gp.noise));
        end
    end
end

