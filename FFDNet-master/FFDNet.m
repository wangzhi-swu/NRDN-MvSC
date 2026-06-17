function [output] = FFDNet(img,imageNoiseSigma)
% This is the testing demo of FFDNet for denoising noisy color images corrupted by
% AWGN.
%
% To run the code, you should install Matconvnet first. Alternatively, you can use the
% function `vl_ffdnet_matlab` to perform denoising without Matconvnet.
%
% "FFDNet: Toward a Fast and Flexible Solution for CNN based Image
% Denoising" 2018/03/23
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)

% clear; clc;

format compact;
global sigmas; % input noise level or input noise level map


folderModel = 'models';
folderTest  = 'testsets';
folderResult= 'results';


showResult  = 1;
useGPU      = 1;
pauseTime   = 0;

inputNoiseSigma = imageNoiseSigma;  % input noise level

load(fullfile('models','FFDNet_gray.mat'));%

net = vl_simplenn_tidy(net);

if useGPU
    net = vl_simplenn_move(net, 'gpu') ;%vl_simplenn_move 将CNN在CPU和GPU之间移动
end


label   = img;
    [w,h,c] = size(label); % 481*321*3

    
    if c == 3 %彩色图
        % add noise
%        randn('seed',0);
        %noise = bsxfun(@times,randn(size(label)),permute(imageNoiseSigma/255,[3 4 1 2]));
        input = single(label);
       
        
        if mod(w,2)==1 %对行取模，如果行数不能被2整除，就把最后一行的切片重复加在input的末尾
            input = cat(1,input, input(end,:,:)) ;
        end
        if mod(h,2)==1%对列取模
            input = cat(2,input, input(:,end,:)) ;
        end
        
        % tic;
        if useGPU
            input = gpuArray(input); %利用gpuArray()函数将数据从CPU传入GPU中
        end

        
        % set noise level map
        sigmas = inputNoiseSigma/255; % see "vl_simplenn.m".
        
        % perform denoising
        res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
        % res    = vl_ffdnet_concise(net, input);    % concise version of vl_simplenn for testing FFDNet
        %res    = vl_ffdnet_matlab(net, input); % use this if you did  not install matconvnet; very slow
        
        % output = input -res(end).x; % for 'model_color.mat'
        output = res(end).x;
        
        
        if mod(w,2)==1 %再把刚才加的最后一行和一列舍去
            output = output(1:end-1,:,:);
            input  = input(1:end-1,:,:);
        end
        if mod(h,2)==1
            output = output(:,1:end-1,:);
            input  = input(:,1:end-1,:);
        end
        
        if useGPU
            output = gather(output); %如果需要的话，用gather()函数将数据从GPU传回CPU
            input  = gather(input);
        end
        %toc;
             
    end
end



