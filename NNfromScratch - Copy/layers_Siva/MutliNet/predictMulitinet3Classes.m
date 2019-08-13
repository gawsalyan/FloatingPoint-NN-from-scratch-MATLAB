 function outputClass = predictMulitinet3Classes(netSiva, X1, X2, X3)
           A1 = netSiva.Nets{1}.predict(netSiva.Nets{1},X1);
           A2 = netSiva.Nets{2}.predict(netSiva.Nets{2},X2);
           A3 = netSiva.Nets{3}.predict(netSiva.Nets{3},X3);
           A = netSiva.Nets{4}.predict(netSiva.Nets{4},...
               [A1{netSiva.Nets{1}.no_ofLayer};A2{netSiva.Nets{2}.no_ofLayer};A3{netSiva.Nets{3}.no_ofLayer}]);
           outputClass = A{end};
 end