function [x] = cappedl1(b,lambda,theta)
u=abs(b);
x1=max(u,theta);
x2=min(theta, max(0, u - lambda));
if (0.5*(x1 + x2 - 2*u)*(x1 - x2) + lambda*(theta - x2) < 0)
    x = x1;
else
    x = x2;
    
end


end