function [Z]=t_SVD(X,lambda1,K,N,theta)
for k=1:K
    Z{k} = zeros(N,N); 
    W{k} = zeros(N,N);
    G{k} = zeros(N,N);
    B{k} = zeros(N,N);
    L{k} = zeros(N,N);
    E{k} = zeros(size(X{k},1),N);
    Y{k} = zeros(size(X{k},1),N); 
end
dim1 = N;dim2 = N;dim3 = K;
myNorm = 'tSVD_1';
sX = [N, N, K];
%set Default
parOP         =    false;
ABSTOL        =    1e-6;
RELTOL        =    1e-4;
Isconverg = 0;epson = 1e-5;
iter = 0;
mu1 = 10e-5; max_mu1 = 10e10; pho_mu1 = 1.6;
mu2 = 10e-5; max_mu2 = 10e10; pho_mu2 = 1.6;
rho = 10e-5; max_rho = 10e10; pho_rho = 1.6;
% mu1 = 10e-5; max_mu1 = 10e10; pho_mu1 = 1.1;
% mu2 = 10e-5; max_mu2 = 10e10; pho_mu2 = 1.1;
% rho = 10e-5; max_rho = 10e10; pho_rho = 2;
D = ones(1,1+round(dim1/2));
while(Isconverg == 0)
    start = 0;
    %-------------------update Z^k-------------------------------
    for k=1:K
        Z{k} = inv(rho*eye(N)+mu1*X{k}'*X{k}) * (X{k}'*(Y{k}+mu1*X{k}-mu1*E{k})-W{k}+rho*G{k});
    end
    %-------------------update E^k-------------------------------
    F = [];
    for k=1:K    
        tmp = X{k}-X{k}*Z{k}+Y{k}/mu1;
        F = [F;tmp];
    end  
    [Econcat] = solve_l1l2(F,lambda1/mu1);
    start = 1;
    for k=1:K
        E{k} = Econcat(start:start + size(X{k},1) - 1,:);
        start = start + size(X{k},1);
    end
    %-------------------update G---------------------------------
    Z_tensor = cat(3, Z{:,:});
    W_tensor = cat(3, W{:,:});
    z = Z_tensor(:);
    w = W_tensor(:); 
    [g, objV] = wshrinkObj(z + 1/rho*w,rho,sX,0,3);
    G_tensor = reshape(g, sX);
%      g = t_CSVT(z + 1/rho*w,D,1/rho,sX);
%      %                 g = t_CSVT1(z + 1/rho*w,D,1/rho,sX,theta);
%      G_tensor = reshape(g, sX);
%      for index = 1:1+round(dim1/2)
%          G1=shiftdim(G_tensor,1);
%          [~,S,~] = svd(G1(:,:,index),'econ');
%          diagS = diag(S);
%          %                       fprintf(' diagS= %.6f\n',diagS);
%          thresh = max(diagS-theta, 0);
%          D(1,index) = size(find(thresh>0),1);
%      end
    %------------------update auxiliary variable---------------
    w = w + rho*(z - g);
    W_tensor = reshape(w, sX);
    for k=1:K
        G{k} = G_tensor(:,:,k);
        Y{k} = Y{k} + mu1*(X{k}-X{k}*Z{k}-E{k});
        W{k} = W_tensor(:,:,k);
    end
    %% coverge condition
    Isconverg = 1;
    for k=1:K
        if (norm(X{k}-X{k}*Z{k}-E{k},inf)>epson)
            history.norm_Z = norm(X{k}-X{k}*Z{k}-E{k},inf);
%             fprintf('    norm_Z %7.10f    ', history.norm_Z);
            Isconverg = 0;
        end
    end
    if (iter>200)
        Isconverg  = 1;
    end
    iter = iter + 1;
    mu1 = min(mu1*pho_mu1, max_mu1);
    mu2 = min(mu2*pho_mu2, max_mu2);
    rho = min(rho*pho_rho, max_rho);
%     fprintf('\n');
end