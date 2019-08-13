function A = skewedToone(X)

maxX = max(X); minX = min(X);

if (maxX*minX <1)
    A = X;
    A(A>0) = X(X>0)/maxX;
    A(A<0) = X(X<0)/abs(minX);
else
    A = (X - 2*minX);
    A(A>0) = A(A>0)/(maxX- minX);
    A(A<0) = A(A<0)/abs(minX);
end
end