function [Z] = FFDNet_demo2(X,sigma)
% This is the testing demo of FFDNet for denoising noisy grayscale images corrupted by
% AWGN with clipping setting. The noisy input is 8-bit quantized.
% "FFDNet: Toward a Fast and Flexible Solution for CNN based Image Denoising"
%  2018/03/23
% If you have any question, please feel free to contact with me.
% Kai Zhang (e-mail: cskaizhang@gmail.com)
pwd='E:\Codes&Datasets\fy\2021TIP_GNLTA_深度\FFDNet-master\utilities';
addpath(pwd);
%  addpath()
%clear; clc;
format compact;
global sigmas; % input noise level or input noise level map

useGPU      = 0;

[w, h, c] = size(X);
num=w*h;

load 'E:\Codes&Datasets\fy\2021TIP_GNLTA_深度\FFDNet-master\models\FFDNet_gray.mat'
% load(fullfile('models','FFDNet_gray.mat'));
net = vl_simplenn_tidy(net);

if useGPU
    net = vl_simplenn_move(net, 'gpu');
end

% if c==3
%     input = X;
%     %sigX = Xmiss;
% else
    %input = unorigami(X,[w h c]);
    input = reshape(X, w*h,c);
    %size(input)
    %sigX = unorigami(Xmiss,[w h c]);
% end

%     sigma = sqrt(lambda/beta2^sqrt(1/r));
%     sigma = max(sigma^sqrt(r),0.02)
%       sigma = 0.1*sqrt(abs(opts.sigma^2-(norm(input(:)-sigX(:))^2)/eleNum));


input = single(input); %
% if c==3
%     if mod(w,2)==1
%         input = cat(1,input, input(end,:,:)) ;
%     end
%     if mod(c,2)==1
%         input = cat(2,input, input(:,end,:)) ;
%     end
% else
    
    if mod(num,2)==1
        input = cat(1,input, input(end,:)) ;
    end
    if mod(c,2)==1
        input = cat(2,input, input(:,end)) ;
    end
% end

if useGPU
    input = gpuArray(input);
end
max_in = max(input(:));min_in = min(input(:));
input = (input-min_in)/(max_in-min_in);

sigmas = sigma/(max_in-min_in);

res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test');
output = res(end).x;

output(output<0)=0;output(output>1)=1;
output = output*(max_in-min_in)+min_in;

% if c==3
%     if mod(w,2)==1
%         output = output(1:end-1,:,:);
%     end
%     if mod(h,2)==1
%         output = output(:,1:end-1,:);
%     end
% else
    if mod(num,2)==1
        output = output(1:end-1,:);
    end
    if mod(c,2)==1
        output = output(:,1:end-1);
    end
% end

if useGPU
    output = gather(output);
end
% if c==3
%     Z = double(output);
% else
    %Z = origami(double(output),[w h c]);
    Z= reshape(double(output),[w h c]);
% end


end

