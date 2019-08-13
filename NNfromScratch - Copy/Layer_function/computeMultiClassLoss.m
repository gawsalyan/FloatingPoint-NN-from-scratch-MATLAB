function L = computeMultiClassLoss(Y, Y_hat, lossname)
   m = size(Y,2);  
   
   if nargin > 2
   L = -(1/m)*sum(Y.*log(Y_hat)+ (1-Y).*log(1-Y_hat),'all');
   else
   L =  sum((Y_hat-Y).^2,'all');  
   end
end
