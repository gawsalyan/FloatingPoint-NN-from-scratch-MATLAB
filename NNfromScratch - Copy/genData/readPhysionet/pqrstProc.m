function [A, Z, Zsum] = pqrstProc(X,m)
Z = zeros(6,252,m);
Zsum = zeros(6,m);
for i = 1:m

%%
x1 = X(:,i)';
% figure(1);
% plot(1:252,x1);
% hold on;
[R , Ri] = max(x1(80:100)); Ri = Ri+79;
[Q , Qi] = min(x1(Ri - 20:Ri)); Qi = Qi + Ri - 21; 
[S , Si] = min(x1(Ri :Ri + 20)); Si = Si + Ri - 1; 
[P , Pi] = max(x1(1 :Qi)); 
[T , Ti] = max(x1(Si :end)); Ti = Ti + Si -1; 

% plot(Pi,P,'og');
% plot(Qi,Q,'^g');
% plot(Ri,R,'or');
% plot(Si,S,'^b');
% plot(Ti,T,'ob');

%%
Pvec = skewedToone(x1(1:Pi));%normalize(x1(1:Pi));
PQvec = skewedToone(x1(Pi:Qi)); %normalize(x1(Pi:Qi));
QRvec = skewedToone(x1(Qi:Ri));
RSvec = skewedToone(x1(Ri:Si));
STvec = skewedToone(x1(Si:Ti));
Tvec = skewedToone(x1(Ti:end));
% figure(2);
% plot(1:Pi, Pvec);
% hold on;
% plot(Pi:Qi, PQvec);
% plot(Qi:Ri, QRvec);
% plot(Ri:Si, RSvec);
% plot(Si:Ti, STvec);
% plot(Ti:252, Tvec);

%% One Hot

Z(1,1:Pi,i) = 1;%Pvec;
Z(2,Pi:Qi,i) = 1;%PQvec;
Z(3,Qi:Ri,i) = 1;%QRvec;
Z(4,Ri:Si,i) = 1;%RSvec;
Z(5,Si:Ti,i) = 1;%STvec;
Z(6,Ti:end,i) = 1;%Tvec;

Zsum(1,i) = sum(Pvec);
Zsum(2,i) = sum(PQvec);
Zsum(3,i) = sum(QRvec);
Zsum(6,i) = sum(RSvec);
Zsum(5,i) = sum(STvec);
Zsum(6,i) = sum(Tvec);


A(:,i) = skewedToone(sum(Z(:,:,i),2));
% figure(3);
% plot(1:252,Z); 
end


end
