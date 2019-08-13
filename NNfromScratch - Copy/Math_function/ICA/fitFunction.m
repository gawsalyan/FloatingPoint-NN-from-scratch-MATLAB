function err = fitFunction(var,funName,params,freeList,origVarargin)
%err = fitFunction(var,funName,params,freeList,origVarargin)
%
%Support function for 'fit.m and fitFunction.m'
%Written by G.M Boynton


%stick values of var into params

params = var2params(var,params,freeList);

%evaluate the function

evalStr = sprintf('err = %s(params',funName);
for i=1:length(origVarargin)
    evalStr= [evalStr,',origVarargin{',num2str(i),'}'];
end
evalStr = [evalStr,');'];
eval(evalStr);
end %function 'fitFunction'


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
end %function 'var2params'