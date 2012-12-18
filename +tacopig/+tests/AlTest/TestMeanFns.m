% A standard test on all Tacopig's mean functions
% The following tests are conducted:
%   *  parse the code with matlab
%   *  instanciate the class with either no arguments or using the code 
%      in the constant string member .teststring
%   *  Check that the npar method returns a value - uses this to make a
%      random test problem with a fixed seed reset so its repeatable
%   *  Check for a 'tacopig:dimMismatch' when X1 and X2 have different 
%      dimensionality in eval(X,par)
%   *  Check that eval enforces the correct number of hyperparameters
%   *  Check the size of the output matrix of eval
%   *  Regression test: ensure eval returns the same matrix as last time
%        - if there is no reference, then save automatically
%        - if it doesn't match, then ask the user whether we overwrite
%   *  Warn if gradients are not functioning
%   *  If gradients are functioning:
%         - numerically check gradients at 10 random operating points

clear all
clear functions
clc

  % Find all the covariance functions
  slash = pwd; slash = slash(1);
  cov_functions = ['+tacopig', slash, '+meanfn',slash,'*.m'];
  filez = dir(cov_functions);
  nmf = length(filez)-1;
  mfn_names = cell(1, nmf);
  offset = 0;
  for i=1:nmf
      fname = filez(i+offset).name(1:end-2);
      if strcmp(fname,'MeanFunc')
        offset = 1;
        fname = filez(i+offset).name(1:end-2);
      end
      mfn_names{i} = fname;
  end
    
  % Compare to previous runs (regression test)
  try
      load regression_test_cache.mat; % regressions
  catch e
      fprintf('No regression tests found >> Creating new cache...\n');
      regressions = [];
  end
  eqthresh = 1e-4; % threshold for being approximately equal
  
    
  for i=1:nmf
      
      % Create instances of these classes
      current = mfn_names{i};
      
      fprintf('Testing %s:', current);
      pass = true;
      
      
      argcount = 0;
      try
         eval(['argcount = nargin(@tacopig.meanfn.', current, ');']);
      catch e
         str = geterr(e);
         fprintf('\n%s\n', str);
         fprintf('Cannot continue testing %s\n',current);
         continue
      end
      currentFull = ['tacopig.meanfn.', current];
      if (argcount ~=0)
          try
            eval(['teststring = ',currentFull,'.teststring;']);
            if (length(teststring) ==0)
                error('failblog');
            end
          catch e
            fprintf(['\nNo Test String for ',current,', but its constructor has arguments!\n']);
            fprintf('Cannot continue testing %s\n',current);
            continue;
          end
            
          fullteststring = ['meanobj = tacopig.meanfn.',teststring,';'];
          try
             eval(fullteststring);
          catch e
             str = geterr(e);
             fprintf('\n%s >> %s\n',fullteststring,e.message);
             fprintf('Cannot continue testing %s\n',current);
             continue
          end
      else
         try
            eval(['meanobj = ', currentFull,'();']); 
         catch e
             fprintf('\n%s\nCannot continue testing %s\n',e.message,current);
             continue;
         end
      end
      
      
        % Check Npar
        rand('seed', 50);
        npoint = 15;
        ndims = 3;
        X  = rand(ndims,npoint);
      
        npar = 0;
        try
            npar = meanobj.npar(ndims);
        catch e
            fprintf('\n%s\nFurther %s tests aborted\n\n', e.message, current);
            continue;
        end
        
        % Look for standard exceptions
        pars = rand(1,npar);
        
        % checking number of hyperparameters
        if ~throws(meanobj, 'k = in.eval(varargin{1}, varargin{2});', 'tacopig:inputInvalidLength', X, [pars,1])
            if (pass)
                fprintf('\n');
            end
            fprintf('%s.eval is not detecting the wrong number of hypers and should throw tacopig:inputInvalidLength\n', current);
            pass = false;
        end     
                
        % check that outputs are of the correct size
        try
            M1 = meanobj.eval(X,pars);
            s = size(M1);
            if ~((s(1)==1)&&(s(2)==npoint))
                error('%s.eval is the wrong size (should be 1*N)!',current);
            end
        catch e
            if (pass)
                fprintf('\n');
            end
            fprintf('%s >> %s\n', current, e.message);
            pass = false;
        end
        
        % and give the same result as before
        try
            eval(sprintf('oldresult = regressions.means.%s.eval;',current));
            if (approxequal(M1,oldresult)>eqthresh)
                if (pass)
                     fprintf('\n');
                end
                fprintf('eval doesn''t match the cached regression data!\n');
                
                reply = input('Are you sure it is correct NOW? Y/N [Y]: ', 's');
                if isempty(reply)
                    reply = 'N';
                end
                if strcmpi(reply,'y')
                    eval(sprintf('regressions.means.%s.eval = M1;',current));
                else
                    pass = false;
                end
            end
        catch e
            fprintf('No previous result for %s.eeval - saving for next time...\n', current);
            eval(sprintf('regressions.means.%s.eval = M1;',current));
        end
        
        % If no gradients then warn
        testgrads = true;
        try
            g = meanobj.gradient(X, pars);
        catch e
            if (pass)
                fprintf('\n');
                pass = false;
            end
            fprintf('%s''s Gradients are not functioning: %s\n',current, e.message);
            testgrads = false;
        end
        
        % If has gradients then check that they actually correspond to working gradients
        if (testgrads&(npar>0))
            
            try
                % check size of output vector
                
                if ((size(g,1)~=1)||(size(g,2)~=npar))
                    error('Gradients should be 1xnpar!');
                end
                
                % Check that analytic is similar to numerical
                eps = 1e-6; % numerical step
                ntestpars = 10;
                
                for p = 1:ntestpars
                    params = rand(1,npar);
                    tX = rand(size(X));
                    
                    g = meanobj.gradient(tX, params);
                    
                    %val0 = meanobj.Keval(X, params);
                    err = false;                 
                    for i = 1:npar
                       params2 = params;
                       params3 = params;
                       params2(i) = params2(i) + eps;
                       params3(i) = params3(i) - eps;
                       val1 = meanobj.eval(tX,params2);
                       val2 = meanobj.eval(tX,params3);
                       numerical = (val1-val2)/(2*eps);
                       analytic  = g{i};
                       if (approxequal(numerical,analytic) > eqthresh)
                            fprintf('Gradient mismatch on parameter %d\n',i);
                            err = true;
                       end
                    end
                    if (err)
                        error('Possible gradient problem!');
                    end
                end

            catch e
                if (pass)
                    fprintf('\n');
                    pass = false;
                end
                fprintf('%s\n',e.message);
            end   
        end
        
        
        % passed everything?
        if (pass)
            fprintf('Passed!\n');
        end
        
        % If not gradients - issue a warning here
        
        fprintf('\n');
  end
  
    % Save the regression tests results as they stand
    try
        save regression_test_cache.mat regressions; % regressions
    catch e
        fprintf('Could not save regression test values... try saving manually!');
    end
  