function[E]=DeCNNAD(X,H,W,K,beta,lamda,net)

    rng(6);
    P=20;
    [D,N]=size(X);
    IDX=kmeans(X',K);




    Dict=[];
    for i=1:K
        pos=find(IDX==i);
        D_temp=X(:,pos);
        if(size(D_temp,2)<10)
            continue;
        end
        for ii=1:3
            mu=mean(D_temp,2);
            COV_inv=pinv(cov(D_temp'));
            D_temp_C=D_temp-repmat(mu,[1,size(D_temp,2)]);
            Dis=zeros(1,size(D_temp,2));
            for j=1:size(D_temp,2)
                Dis(j)=D_temp_C(:,j)'*COV_inv*D_temp_C(:,j);
            end
            [Val,Ind]=sort(Dis);
            quan=floor(size(Ind,2)*0.70);
            tmp=D_temp(:,Ind(1:quan));
            clearD_temp;
            D_temp=tmp;
        end

        if(size(D_temp,2)<P)
            continue;
        else
            Dict=[Dict,D_temp(:,(1:P/2)),D_temp(:,(end-P/2+1:end))];

        end

    end



    global sigmas




    net=vl_simplenn_move(net,'gpu');




    display=1;

    tol1=1e-6;
    tol2=1e-6;
    maxIter=50;
    mu=0.01;
    mu1=1;
    mu_max=1e10;
    ita1=1.0/((norm(Dict,2))^2);

    [dim,num]=size(X);
    numDict=size(Dict,2);
    S=zeros(numDict,num);

    DtX=Dict'*X;
    DtD=Dict'*Dict;
    E=zeros(dim,num);
    Y1=zeros(numDict,num);
    Z=zeros(numDict,num);


    iter=0;
    X_F=norm(X,'fro');




    Echange=zeros(1,maxIter);
    while iter<maxIter
        iter=iter+1;

        temp=Dict'*(X-E)+mu*Z+Y1;
        S1=inv(DtD+mu1*eye(numDict))*temp;



        tmp=S1-Y1/mu;
        N_Img=reshape(tmp',H,W,numDict);










        BM3D_Img=N_Img;






        for jj=1:size(N_Img,3)
            eigen_im=(N_Img(:,:,jj));
            min_x=min(eigen_im(:));
            max_x=max(eigen_im(:));
            eigen_im=eigen_im-min_x;
            scale=max_x-min_x;
            eigen_im=(eigen_im/scale);
            sigmas=beta/mu/scale;

            input=gpuArray(eigen_im);
            res=vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');




            BB=gather(res(end).x);
            BM3D_Img(:,:,jj)=double(BB)*scale+min_x;
        end
        Z1=reshape(BM3D_Img,num,numDict)';


        temp=X-Dict*S1;
        E1=solve_l1l2(temp,lamda);

        RES=X-Dict*S1-E1;
        Y1=Y1+mu*(Z1-S1);

        ktt1=norm(RES,'fro')/X_F;

        ktt3=norm(E1-E,'fro')/X_F;

        ktt2=max(norm(S1-S,'fro'),norm(E1-E,'fro'))/X_F;





        rou=1.1;
        mu=min(mu_max,rou*mu);

        S=S1;
        E=E1;
        Z=Z1;
        Echange(1,iter)=ktt1;
        if display
            disp(['iter ',num2str(iter)]);



        end
        if(ktt1<tol1&&ktt2<tol2)

        end
    end

end
