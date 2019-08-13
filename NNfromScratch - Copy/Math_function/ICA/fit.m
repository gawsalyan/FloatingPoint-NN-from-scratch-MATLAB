function [params,err] = fit(funName,params,freeList,varargin)
%[params,err] = fit(funName,params,freeList,var1,var2,var3,...)
%
%Helpful interface to matlab's 'fminsearch' function.
%
%INPUTS
% 'funName':  function to be optimized.  Must have form err = <funName>(params,var1,var2,...)
% params   :  structure of parameter values for fitted function
%     params.options :  options for fminsearch program (see OPTIMSET)
% freeList :  Cell array containing list of parameter names (strings) to be free in fi
% var<n>   :  extra variables to be sent into fitted function
%
%OUTPUTS
% params   :  structure for best fitting parameters 
% err      :  error value at minimum
%
%See 'FitDemo.m' for an example.
%
%Written by Geoffrey M. Boynton, Summer of '00


%turn free parameters in to 'var'
if isfield(params,'options')
  options = params.options;
else
  options = [];
end


if isempty(freeList)
  freeList = fieldnames(params);
end

vars = params2var(params,freeList);

if ~isfield(params,'shutup')
  disp(sprintf('Fitting "%s" with %d free parameters.',funName,length(vars)));
end

vars = fminsearch('fitFunction',vars,options,funName,params,freeList,varargin);

%get final parameters
params=  var2params(vars,params,freeList);

%evaluate the function

evalStr = sprintf('err = %s(params',funName);
for i=1:length(varargin)
  evalStr= [evalStr,',varargin{',num2str(i),'}'];
end
evalStr = [evalStr,');'];
eval(evalStr);

end


function var = params2var(params,freeList)
%var = params2var(params,freeList)
%
%Support function for 'fit.m'
%Written by G.M Boynton, Summer of '00

var = [];
for i=1:length(freeList)
  evalStr = sprintf('tmp = params.%s;',freeList{i});
  eval(evalStr);
  var = [var,tmp(:)'];
end
end

function params = var2params(var,params,freeList)
%params = var2params(var,params,freeList)
%
%Support function for 'fit.m'
%Written by G.M Boynton, Summer of '00

count = 1;
for i=1:length(freeList)
  evalStr = sprintf('len = length(params.%s);',char(freeList(i)));
  eval(evalStr);
  evalStr = sprintf('params.%s =  var([%d:%d]);',char(freeList(i)),count,count+len-1);
  eval(evalStr);
  count = count+len;
end
end









