clc;clear
addpath('./ClusteringMeasure');
addpath('./code_coregspectral');
addpath('./database')
addpath('./LRR');
addpath('./twist');
addpath('./FFDNet-master');



%% UCI
Dataname    = 'uci_digit.mat';
load(Dataname);

cls_num    = length(unique(gt));

%% Note: each column is an sample (same as in LRR)
X{1}=X1;
X{2}=X2;
X{3}=X3;
for v=1:3
    [X{v}]=NormalizeData(X{v});
end
N = size(X{1},2); %sample number
V= length(X);
tem_Z = zeros(N,N);
dim1 = N;dim2 = N;dim3 = V;
myNorm = 'tSVD_1';
sX = [N, N, V];
%set Default
parOP         =    false;
ABSTOL        =    1e-6;
RELTOL        =    1e-4;
threshold=3;
tol=1e-6;

veclambda    =0.0001;
veclambda1    =1e-7;
vectheta    =0.0000001;
vectheta1    =1;

for index1  = 1:length(veclambda)
    lambda = veclambda(index1);
    for index2  = 1:length(veclambda1)
        lambda1 = veclambda1(index2);
        for index3  = 1:length(vectheta)
            theta = vectheta(index3);
            for index4 = 1:length(vectheta1)
                theta1 = vectheta1(index4);
                fprintf('lambda= %.10f  ', lambda);
                fprintf('lambda1= %.10f  ', lambda1);
                fprintf('theta= %.10f  ', theta);
                fprintf('theta1= %.8f  ', theta1);
                fprintf('\n');
                ACC = [];
                NMI = [];
                F=[];
                P=[];
                R=[];
                AR=[];
                for t=1:1
                    t1=clock;
                    epson = 1e-6;
                    start = 1;
                    iter   = 0;
                    mu     = 10e-5;
                    max_mu = 10e10;
                    pho_mu = 2;
                    Isconverg = 0;
                    D = ones(1,1+round(dim1/2));
                    for k=1:V
                        Z{k} = zeros(N,N);
                        G{k} = zeros(N,N);
                        Y{k} = zeros(N,N);
                        Y1{k} = zeros(size(X{k},1),N);
                        Y2{k} = zeros(N,N);
                        Y3{k} = zeros(N,N);
                        E{k} = zeros(size(X{k},1),N);
                    end
                    Z_tensor = cat(3, Z{:,:});
                    G_tensor = cat(3, G{:,:});
                    Y_tensor = cat(3, Y{:,:});
                    Error1=[];
                    ACC_New=[];
                    NMI_New=[];
                    Time_New=[];
                    while (Isconverg == 0)
                        Zpre=Z_tensor;
                        %                         Epre=E_tensor;
                        Gpre=G_tensor;
                        Ypre=Y_tensor;
                        %% update Z
                        for k=1:V
                            tmp=(1/mu)*X{k}'*Y1{k}+X{k}'*X{k}-X{k}'*E{k}-(1/mu)*(Y2{k}+Y3{k})+G{k}+Y{k};
                            Z{k} =inv(X{k}'*X{k}+2*eye(N,N))*tmp;
                        end
                        Z_tensor = cat(3, Z{:,:});
                        
                        %% update E
                        F = [];
                        for k=1:V
                            tmp = X{k}-X{k}*Z{k}+Y1{k}/mu;
                            F = [F;tmp];
                        end
                        F=F';
                        norm_temp_T=sqrt(sum(F.^2,2));
                        shrink_norm=C_2_23(norm_temp_T,lambda/mu,theta1);
                        [m n] = size(F);
                        for i=1:m
                            if shrink_norm(i)==0
                                E_hat(i,:)=zeros(1,n);
                            else
                                E_hat(i,:)=F(i,:)*shrink_norm(i)/norm_temp_T(i);
                            end
                        end
                        E_hat=E_hat';
                        start = 1;
                        for k=1:V
                            E{k} = E_hat(start:start + size(X{k},1) - 1,:);
                            start = start + size(X{k},1);
                        end
                        %                         E_tensor = cat(3, E{:,:});
                        clear E_hat F
                        
                        %% update G
                        Y2_tensor = cat(3, Y2{:,:});
                        y2 = Y2_tensor(:);
                        z=Z_tensor(:);
                        g = t_SC23(z + 1/mu*y2,1/mu,sX,theta);
                        G_tensor = reshape(g, sX);
                        for i=1:V
                            G{i}=G_tensor(:,:,i);
                        end
                        
                        %% update Y
                        Y3_tensor=cat(3, Y3{:,:});
                        temp =Z_tensor+ Y3_tensor/mu;
                        Y_tensor=FFDNet_demo2(temp,sqrt(lambda1/mu));
                        for i=1:V
                            Y{i}=Y_tensor(:,:,i);
                        end
                        %% update multipliers
                        for k=1:V
                            Y1{k} = Y1{k} + mu*(X{k}-X{k}*Z{k}-E{k});
                            Y2{k} = Y2{k} + mu*(Z{k}-G{k});
                            Y3{k} = Y3{k} + mu*(Z{k}-Y{k});
                        end
                        Y2_tensor =Y2_tensor+mu*(Z_tensor-G_tensor);
                        Y3_tensor = Y3_tensor+mu*(Z_tensor-Y_tensor);
                        %% check convergence
                        maxleq=0;
                        for k=1:V
                            leq = X{k}-X{k}*Z{k}-E{k};
                            leq1 = max(abs(leq(:)));
                            leqm1=max(leq1,maxleq);
                            maxleq=leqm1;
                        end
                        
                        
                        leq2 = Z_tensor-G_tensor;
                        leq3 = Z_tensor-Y_tensor;
                        leqm2 = max(abs(leq2(:)));
                        leqm3 = max(abs(leq3(:)));
                  
                         err = max([leqm1,leqm2]);
                       % Different datasets have varying convergence speeds and complexities, therefore the the maximum number of iterations may differ.
                        if err < tol || iter>20 
                            Isconverg  = 1;
                        end
                        iter = iter + 1;
                        mu = min(mu*pho_mu, max_mu);
                    end
                    S = zeros(N,N);
                    for k=1:V
                        S = S + abs(Z{k})+abs(Z{k}');
                    end
                    S=S/(2*V);

                    C= SpectralClustering(S,cls_num);
                    acc = Accuracy(C,double(gt));
                    [A, nmi ,avgent] = compute_nmi(gt,C);
                    [f,p,r] = compute_f(gt,C);
                    [ar,ri,MI,HI]=RandIndex(gt,C);
                    result_ACC(t)= acc;
                    result_NMI(t)=nmi;
                    result_F(t)=f;
                    result_P(t)=p;
                    result_R(t)=r;
                    result_AR(t)=ar;
                    t2 = clock;
                    Time = etime(t2,t1);
                    ACC=[mean(result_ACC),std(result_ACC)];
                    NMI=[mean(result_NMI),std(result_NMI)];
                    F=[mean(result_F),std(result_F)];
                    P=[mean(result_P),std(result_P)];
                    R=[mean(result_R),std(result_R)];
                    AR=[mean(result_AR),std(result_AR)];
                end
                fprintf('ACC = %.6f (%.6f), NMI = %.6f (%.6f),AR = %.6f (%.6f)\n',ACC(1),ACC(2),NMI(1),NMI(2),AR(1),AR(2));
                fprintf('F = %.6f (%.6f), P = %.6f (%.6f), R = %.6f (%.6f)\n',F(1),F(2),P(1),P(2),R(1),R(2));
                fprintf('Time: %10.4f\n', Time);
                fprintf('--------------------------------------\n');
                dlmwrite('result_UCI.txt', [ lambda, lambda1,theta,theta1,ACC(1),NMI(1), AR(1), F(1),P(1), R(1)] , '-append', 'delimiter', '\t', 'newline', 'pc');
            end
        end
        
    end
end




