function delta = C_2_23(norm_temp_T,lambda,tau)

%     [m,n] = size(G);
%     sigma = diag(Sigma);
delta = zeros(size(norm_temp_T));
%% self
%     tau = opt.tau;
v = (2*lambda*(1-(2/3)))^(1.0/(2-(2/3)));
v1 = v + lambda*(2/3)*v^((2/3)-1);
for j=1:size(norm_temp_T)
    s = norm_temp_T(j);
     x_ = solve_l23(2*lambda,s);
    tau_ = ((1.0/(2*lambda))*(x_-s)^2 + x_^(2/3))^(1.0/(2/3));
    if tau <= tau_
        delta(j) = s;
    else
        delta(j) = x_;
    end
end

