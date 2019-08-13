 function outputClass = predictTransfernetClasses(netSiva, X0, X1, X2)
           A1 = predictMulitinetClasses(netSiva.Nets{1},X0, X1);
           A2 = netSiva.Nets{2}.predict(netSiva.Nets{2},X2);
           A = netSiva.Nets{3}.predict(netSiva.Nets{3},...
               [A1;A2{netSiva.Nets{2}.no_ofLayer}]);
           outputClass = A{end};
 end