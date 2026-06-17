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
addpath(fullfile('utilities'));

folderModel = 'models';
folderTest  = 'testsets';
folderResult= 'results';
imageSets   = {'CBSD68','Kodak24','McMaster'}; % testing datasets
setTestCur  = imageSets{1};      % current testing dataset

showResult  = 1;
useGPU      = 0;
pauseTime   = 0;

imageNoiseSigma = 25;  % image noise level
inputNoiseSigma = 25;  % input noise level

folderResultCur       =  fullfile(folderResult, [setTestCur,'_',num2str(imageNoiseSigma(1)),'_',num2str(inputNoiseSigma(1))]);
if ~isdir(folderResultCur)
    mkdir(folderResultCur)
end

load(fullfile('models','FFDNet_color.mat'));%fullfile函数作用是作用是利用文件各部分信息创建并合并成完整文件名：
%load net ：这是预先训练好的一个网络
%将其变为simplenn的网络
%matconvnet有两种网络：还有一种为DAG 模型，
% 两个网络的不同之处在于将网络以不同的形式显示出来，后者DAG 会更直观
net = vl_simplenn_tidy(net);

% for i = 1:size(net.layers,2)
%     net.layers{i}.precious = 1;
% end

if useGPU
    net = vl_simplenn_move(net, 'gpu') ;%vl_simplenn_move 将CNN在CPU和GPU之间移动
end

% read images
ext         =  {'*.jpg','*.png','*.bmp','*.tif'};
filePaths   =  [];
for i = 1 : length(ext)
    filePaths = cat(1,filePaths, dir(fullfile(folderTest,setTestCur,ext{i})));
    %cat(1, A, B)相当于[A; B].
end
%把./testsets/CBSD68/文件夹下，所有满足后缀要求的都放在filePaths中。

% PSNR and SSIM
PSNRs = zeros(1,length(filePaths));
SSIMs = zeros(1,length(filePaths));

for i = 1:length(filePaths)%对每一张图片的遍历 length(filePaths)=68
    
    % read images
    %          imread ='testsets\CBSD68\101085.png'
    label   = imread(fullfile(folderTest,setTestCur,filePaths(i).name));
    [w,h,c] = size(label); % 481*321*3

    
    if c == 3 %彩色图
        [~,nameCur,extCur] = fileparts(filePaths(i).name); 
        %fileparts 该函数用于将一个文件的完整路径中各部分提取出来。这一步有啥用呢
        label = im2double(label);
        
        % add noise
        randn('seed',0);
        noise = bsxfun(@times,randn(size(label)),permute(imageNoiseSigma/255,[3 4 1 2]));
        %permute：重新排列矩阵的维度，把3 4维提前
        input = single(label + noise);
        %如果您有一个数组由不同的数据类型（如 double 或 int8）组成，则可以使用 single 函数将该数组转换为单精度数组。
        
        if mod(w,2)==1 %对行取模，如果行数不能被2整除，就把最后一行的切片重复加在input的末尾
            input = cat(1,input, input(end,:,:)) ;
        end
        if mod(h,2)==1%对列取模
            input = cat(2,input, input(:,end,:)) ;
        end
      %最终效果  size(input)= 482   322     3
        % tic;
        if useGPU
            input = gpuArray(input); %利用gpuArray()函数将数据从CPU传入GPU中
        end
        %数据在进行运算时，只要有一个变量在GPU上，其他变量也会自动进入 GPU一起运算，产生的结果也在GPU上。
        %如果需要的话，用gather()函数将数据从GPU传回CPU
        
        % set noise level map
        sigmas = inputNoiseSigma/255; % see "vl_simplenn.m".
        
        % perform denoising
        %res    = vl_simplenn(net,input,[],[],'conserveMemory',true,'mode','test'); % matconvnet default
        % res    = vl_ffdnet_concise(net, input);    % concise version of vl_simplenn for testing FFDNet
        res    = vl_ffdnet_matlab(net, input); % use this if you did  not install matconvnet; very slow
        
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
        
        % calculate PSNR, SSIM and save results
        [PSNRCur, SSIMCur] = Cal_PSNRSSIM(im2uint8(label),im2uint8(output),0,0);
        if showResult
            imshow(cat(2,im2uint8(input),im2uint8(label),im2uint8(output)));
            title([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
            %imwrite(im2uint8(output), fullfile(folderResultCur, [nameCur, '_' num2str(imageNoiseSigma(1),'%02d'),'_' num2str(inputNoiseSigma(1),'%02d'),'_PSNR_',num2str(PSNRCur*100,'%4.0f'), extCur] ));
            drawnow;
            pause()
        end
        disp([filePaths(i).name,'    ',num2str(PSNRCur,'%2.2f'),'dB','    ',num2str(SSIMCur,'%2.4f')])
        PSNRs(i) = PSNRCur;
        SSIMs(i) = SSIMCur;
        
    end
end

disp([mean(PSNRs),mean(SSIMs)]);




