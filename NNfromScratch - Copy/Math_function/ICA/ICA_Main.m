clc
%signal
X = [1 1 2 0 5 4 5 3;...
     3 2 3 3 4 5 5 4];

n = size(X,2); % signal lenth

%% Centering
Mu = mean(X,2);
D = X - Mu;

%% The whitening data step
%Covariance Mat
E = expectedValue(D,n)
%Eigen Vectors and valuse
[V, lambda] = eig(E);   %check orthogonal V(1,:)*V(2,:)' = 0

%%Decorelation
%%The two signals are decorrelated by projecting the centered dataonto the PCA space as follows, U = VD.
U = V*D; % 
E_U = expectedValue(U,n) %check whether decorreleated

%%Resacling to unit variance
Z = ((lambda)^(-0.5))*U
E_Z = expectedValue(Z,n) %The covariance matrix for the whitened data









%%






