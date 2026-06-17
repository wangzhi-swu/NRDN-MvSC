function [output] = FFDNet_patch(patch,imageNoiseSigma)
% This is the testing demo of FFDNet for denoising noisy grayscale images corrupted by
% AWGN with clipping setting. The noisy input is 8-bit quantized.
% "FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising"
%  2018/03/23
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)

%clear; clc;
format compact;
global sigmas; % input noise level or input noise level map

useGPU      = 1; % CPU or GPU. For single-threaded (ST) CPU computation, use "matlab -singleCompThread" to start matlab.

%imageNoiseSigma = 25;  % image noise level, 25.5 is the default setting of imnoise( ,'gaussian')
inputNoiseSigma = imageNoiseSigma;  % input noise level

load(fullfile('models','FFDNet_gray.mat'));
net = vl_simplenn_tidy(net);

% for i = 1:size(net.layers,2)
%     net.layers{i}.precious = 1;
% end

if useGPU
    net = vl_simplenn_move(net, 'gpu');
end

    
    % read images
    input1 = patch;
    [w,h,~]=size(input1);

    if mod(w,2)==1
        input1 = cat(1,input1, input1(end,:));
    end
    if mod(h,2)==1
        input1 = cat(2,input1, input1(:,end)) ;
    end
    
    % tic;
    if useGPU
        input1 = gpuArray(input1);
    end
    
    % set noise level map
    sigmas = inputNoiseSigma/255; % see "vl_simplenn.m".
    
    % denoising
    %res    = vl_ffdnet_matlab(net, input1);
    res    = vl_simplenn(net,input1,[],[],'conserveMemory',true,'mode','test');
    output = res(end).x;
    
    
    if mod(w,2)==1
        output = output(1:end-1,:);
        input1  = input1(1:end-1,:);
    end
    if mod(h,2)==1
        output = output(:,1:end-1);
        input1  = input1(:,1:end-1);
    end
    
    if useGPU
        output = gather(output);
        input1  = gather(input1);
    end


end

