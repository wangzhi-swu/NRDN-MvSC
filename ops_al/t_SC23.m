function z = t_SC23(g,lambda,sX,tau)
%     lambda = opt.lambda;
G = reshape(g,sX);
G=shiftdim(G, 1);
%     size(G)
%G = X + lambda*Z;
G = fft(G,[],3);
[n1,n2,n3] = size(G);
Z = zeros(n1,n2,n3);

%     theta = mu;
[Q,Sigma,R] = svd(G(:,:,1),'econ');
p = 2/3;
%     [m,n] = size(G);
sigma = diag(Sigma);
delta = zeros(size(sigma));
%% self
%     tau = opt.tau;
v = (2*lambda*(1-(2/3)))^(1.0/(2-(2/3)));
v1 = v + lambda*(2/3)*v^((2/3)-1);
for j=1:size(sigma)
    s = sigma(j);
    x_ = solve_l23(2*lambda,s);
    tau_ = ((1.0/(2*lambda))*(x_-s)^2 + x_^(2/3))^(1.0/(2/3));
    if tau <= tau_
        delta(j) = s;
    else
        delta(j) = x_;
    end
end
Delta = zeros(size(Sigma));
Delta(1:size(sigma),:) = diag(delta);
Z(:,:,1)= Q*Delta*R';


% i=2,...,halfn3
halfn3 = round(n3/2);
for i=2 : halfn3
    [Q,Sigma,R] = svd(G(:,:,i),'econ');
    sigma = diag(Sigma);
    delta = zeros(size(sigma));
    v = (2*lambda*(1-(2/3)))^(1.0/(2-(2/3)));
    v1 = v + lambda*(2/3)*v^((2/3)-1);
    for j=1:size(sigma)
        s = sigma(j);
        x_ = solve_l23(2*lambda,s);
        tau_ = ((1.0/(2*lambda))*(x_-s)^2 + x_^(2/3))^(1.0/(2/3));
        if tau <= tau_
            delta(j) = s;
        else
            delta(j) = x_;
        end
    end
    Delta = zeros(size(Sigma));
    Delta(1:size(sigma),:) = diag(delta);
    Z(:,:,i) = Q*Delta*R';
    Z(:,:,n3+2-i) = conj(Q)*Delta*conj(R)';
end

if mod(n3,2) == 0
    i = halfn3+1;
    [Q,Sigma,R] = svd(G(:,:,i),'econ');
    sigma = diag(Sigma);
    delta = zeros(size(sigma));
    %% self
    %     tau = opt.tau;
    v = (2*lambda*(1-(2/3)))^(1.0/(2-(2/3)));
    v1 = v + lambda*(2/3)*v^((2/3)-1);
    for j=1:size(sigma)
        s = sigma(j);
        x_ = solve_l23(2*lambda,s);
        tau_ = ((1.0/(2*lambda))*(x_-s)^2 + x_^(2/3))^(1.0/(2/3));
        if tau <= tau_
            delta(j) = s;
        else
            delta(j) = x_;
        end
    end
    Delta = zeros(size(Sigma));
    Delta(1:size(sigma),:) = diag(delta);
    Z(:,:,i)= Q*Delta*R';
end
Z = ifft(Z,[],3);
Z = shiftdim(Z, 2);
z = Z(:);
end

