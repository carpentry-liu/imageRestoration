close all; clear all; clc;
%% 通过位置信息来产生高斯噪声
% I = imread ('input/2.bmp');
% I = im2double(I);
% V = zeros(size(I));                 %建立矩阵V
% for i = 1:size(V, 1)
%     V(i, :) = 0.02*i/size(V, 1);
% end
% J = imnoise(I, 'localvar', V);      %添加高斯噪声
% figure,
% subplot(121), imshow(I);
% subplot(122), imshow(J); 
% %% 通过亮度值来产生高斯噪声
% I = imread ('input/1.jpg');
% I = im2double(I);
% h = 0:0.1:1;
% v = 0.01:-0.001:0;
% J = imnoise (I, 'localvar', h, v);      %添加噪声
% figure,
% subplot(121), imshow(I);
% subplot(122), imshow(J);
%% 添加椒盐噪声
% % 噪声密度为d， 即噪声占整个像素总数的百分比，系统默认0.05
% % salt & pepper , 符号&前后必须有空格
% % 负脉冲以黑点（胡椒点）出现在图像中
% % 正脉冲以白点（盐点）出现在图像中
% % 去除椒盐噪声比较好的方法是中值滤波
% I = imread ('input/1.jpg');
% I = im2double(I);
% %J = imnoise(I, 'salt & pepper', 0.01);     %添加椒盐噪声
% %K = imnoise(I, 'salt & pepper', 0.03);
% % R = rand(size(I));
% % J = I;
% % J (R<=0.02) = 0;                            %添加椒噪声
% % K = I;
% % K (R<=0.03) = 1;                            %添加盐噪声
% J = imnoise (I, 'poisson');                   %添加泊松噪声
% figure,
% subplot(121), imshow (I);
% subplot(122), imshow (J);
%% 添加乘性噪声 J = imnoise (I, 'speckle', v)
% % J = I * n * I , n 为均值为0、方差为v的均匀分布的随机噪声, 默认值0.04
% I = imread ('input/2.bmp');
% J = imnoise (I, 'speckle');             %添加乘性噪声
% K = imnoise (I, 'speckle', 0.2);
% figure,
% subplot(121), imshow (J);
% subplot(122), imshow (K);
% %% 产生均匀分布噪声
% m = 256; n = 256;
% a = 50; b = 180;
% I = a + (b  - a) * rand (m, n);         %均匀分布噪声
% figure,
% subplot(121), imshow(uint8(I));         %显示噪声图像
% subplot(122), imhist(uint8(I));
% %% 产生指数分布噪声
% m = 256; n = 256;
% a = 0.04;
% K = -1/a;
% I = K*log(1-rand(m,n));                 %指数分布噪声
% figure,
% subplot(121), imshow(uint8(I));         %显示噪声图像
% subplot(122), imhist(uint8(I));         %显示直方图
%% 对图像进行算术均值和几何均值滤波
% % fspecial函数用于建立预定义的滤波算子
% % h = fspecial(type) h = fspecial(type, para)
% I = imread ('input/2.bmp');
% I = im2double(I);
% I = imnoise (I, 'gaussian', 0, 0.05);       %均值为0，方差为0.05的高斯噪声。均值默认值为0，方差 0.01
% PSF = fspecial ('average', 3);              %产生PSF
% J = imfilter(I, PSF);                       %算术均值滤波
% K = exp(imfilter(log(I), PSF));             %几何均值滤波
% figure,
% subplot(131), imshow(I);                    %显示有噪声的图像
% subplot(132), imshow(J);
% subplot(133), imshow(K);
%% 逆谐波均值滤波器
% I = imread ('input/2.bmp');
% I = im2double(I);
% I = imnoise (I, 'salt & pepper', 0.01);     %添加椒盐噪声
% PSF = fspecial ('average', 3);              %产生PSF
% Q1 = 1.6; Q2 = -1.6;
% j1 = imfilter(I.^(Q1+1), PSF);
% j2 = imfilter(I.^Q1, PSF);
% J = j1./j2;                                  %逆谐滤波，Q为正，去除椒噪声(黑点）
% K1 = imfilter(I.^(Q2+1), PSF);
% K2 = imfilter(I.^Q2, PSF);
% K = K1./K2;                                  %逆谐滤波，Q为负，去除盐噪声（白点）
% figure,
% subplot(131), imshow(I);
% subplot(132); imshow(J);
% subplot(133); imshow(K);
%% 顺序统计滤波。中值滤波能很好保留图像边缘，适合去除椒盐噪声，效果优于均值滤波。
% % 最大值滤波也能去除椒盐噪声，但会从黑色边缘去除一些黑色素。
% % 同理  最小值滤波，会从白色物体边缘去除一些白色素。
% % J = ordfilt2(I, order, domain):
% %对图像进行二维排序滤波，矩阵domain中非0值进行排序，order为选择像素位置
% I = imread ('input/2.bmp');
% I = rgb2gray(I);
% I = im2double(I);
% I = imnoise (I, 'salt & pepper', 0.05);         %添加椒盐噪声， 系统默认0.05
% J = medfilt2(I, [3, 3]);                        %二维中值滤波
% domain = [0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0];     %窗口模板
%J = ordfilt2(I, 6, domain);        
%J = ordfilt2(I, 1, ones(4, 4));                   %最大值滤波
%J = ordfilt2(I, 9, ones(3));                      %最小值滤波
% figure;
% subplot (121);imshow (I);
% subplot (122);imshow (J);
%% 自适应维纳滤波
% % J = wiener2(I, [m, n], noise):采用窗口m*n 默认3*3。noise为噪声的能量
% % [J, noise] = wiener2(I, [m, n]):  对噪声进行估计，noise为噪声的能量
% % imcrop(I, [a, b, c, d]);  (a,b)表示裁剪后左上角像素在原图像中的位置，
% % c表示裁剪后的宽，d表示裁剪后的高
% I = imread ('input/2.bmp');
% I = rgb2gray(I);
% I = imcrop(I, [100, 100, 1024, 1024]);          %图像剪切
% J = imnoise(I, 'gaussian', 0, 0.03);               %添加噪声
% [K, noise] = wiener2(J, [5, 5]);                %自适应滤波
% figure,
% subplot(121), imshow(J), axis on;
% subplot(122), imshow(K), axis on;
%% 逆滤波器对图像进行复原
% [X, Y] = meshgrid(x, y) 
% 输出X的每一行的数值都是复制的x的值；输出Y的每一列都的数值都是复制y的值
% imshow(I, []) 相当于imshow[min(A(:)),max(A(:))],使用最小值作为黑色，最大值作为白色。
% I = imread ('input/1.jpg');
% I = im2double(I);
% [m, n] = size (I);
% M = 2*m; N = 2*n;
% u = -m/2:m/2-1;
% v = -n/2:n/2-1;
% [U, V] = meshgrid(u, v);
% D = sqrt (U.^2 + V.^2);
% D0 = 130;                          %截止频率
% H = exp(-(D.^2)./(2*(D0^2)));       %高斯低通滤波器
% N = 0.01*ones(size(I, 1), size(I, 2));
% N = imnoise(N, 'gaussian', 0, 0.001);%添加噪声
% J = fftfilter(I, H) + N;              %频域滤波并加入噪声
% figure,
% subplot(121), imshow(I);               %显示原始图像
% subplot(122), imshow(J, []);            %显示退化后的图像
% HC = zeros(m, n);
% M1 = H > 0.1;                           %频率范围
% HC(M1) = 1./H(M1);
% K = fftfilter(J, HC);                   %逆滤波
% HC = zeros(m, n);
% M2 = H > 0.01;
% HC(M2) = 1./H(M2);
% L = fftfilter(J, HC);                   %进行逆滤波
% figure,
% subplot (121), imshow (K, []);          %显示结果
% subplot (122), imshow (L, []);
%% 维纳滤波复原
% J = deconvwnr(I, PSF, NSR) PSF为点扩展函数，NSR为信噪比
% J = deconvwnr(I, PSF, NCORR, ICORR) NCORR为噪声的自相关函数，ICORR为原始图像自相关函数
% I = imread('input/1.jpg');
% I = rgb2gray(I);
% I = im2double(I);
% LEN = 25;                   %参数设置
% THETA = 20;
% PSF = fspecial('motion', LEN, THETA);   %产生PSF，运动位移为25个像素，角度为20度
% J = imfilter(I, PSF, 'conv', 'circular');%运动模糊
% NSR = 0;
% K = deconvwnr(J, PSF, NSR);             %维纳滤波复原
% figure,
% subplot(131), imshow(I);
% subplot(132), imshow(J);
% subplot(133), imshow(K);
%% 维纳滤波对含有噪声的运动模糊图像进行复原
% I = imread ('input/1.jpg');
% I = rgb2gray(I);
% I = im2double(I);
% LEN = 21;
% THETA = 11;
% PSF = fspecial('motion', LEN, THETA);
% J = imfilter(I, PSF, 'conv', 'circular');           %产生运动模糊
% noise_mean = 0;
% noise_var = 0.0001;
% K = imnoise(J, 'gaussian', noise_mean, noise_var)  %添加高斯噪声
% figure,
% subplot(121), imshow(I);
% subplot(122), imshow(K);
% NSR1 = 0;
% L1 = deconvwnr(K, PSF, NSR1);                   %维纳滤波复原
% NSR2 = noise_var/var(I(:));
% L2 = deconvwnr(K, PSF, NSR2);           
% figure,
% subplot(121), imshow(L1);
% subplot(122), imshow(L2);
%% 自相关信息进行复原
% I = imread('input/2.bmp');
% I = im2double(I);
% LEN = 20;                                       %设置参数
% THETA = 10;
% PSF = fspecial('motion', LEN, THETA);           %产生PSF
% J = imfilter(I, PSF, 'conv', 'circular');        %运动模糊
% figure,
% subplot(121), imshow(I);
% subplot(122), imshow(J); 
% noise = 0.03*randn(size(I));
% K = imadd(J, noise);                        %添加噪声
% NP = abs(fft2(noise)).^2;                   
% NPower = sum(NP(:))/prod(size(noise));
% NCORR = fftshift(real(ifft2(NP)));          %噪声的自相关函数
% IP = abs (fft2(I)).^2;
% IPower = sum(IP(:))/prod(size(I));
% ICORR = fftshift(real(ifft2(IP)));          %图像的自相关函数
% L = deconvwnr(K, PSF, NCORR, ICORR);        %图像复原
% figure,
% subplot(121), imshow(K);                    %显示结果
% subplot(122), imshow(L);                    %显示结果
%% 约束最小二乘法复原
% J = deconvreg(I, PSF, NOISEPOWER， LRANGE, REGOP) 
% PSF 为点扩展函数 NOISEPOWER为噪声强度，默认值为0 LRANGE为拉格朗日算子搜索范围，默认值为[10^-9, 10^9]
% REGOP 为约束算子
% [J, LAGRA] = deconvreg(I, PSF,...)  返回值LAGRE为最终采用的拉个朗日算子
% I = imread ('input/2.bmp');
% I = im2double(I);
% PSF = fspecial('gaussian', 8, 4);       %产生PSF
% J = imfilter(I, PSF, 'conv');           %图像退化
% figure,
% subplot(121), imshow(I);
% subplot(122), imshow(J);                %显示退化后的图像
% v = 0.02;
% K = imnoise (J, 'gaussian', 0, v);      %添加噪声
% NP = v*prod(size(I));
% L = deconvreg(K, PSF, NP);              %图像复原
% figure,
% subplot(121), imshow(K);
% subplot(122), imshow(L);
%% 拉格朗日算子进行图像复原
% I = imread('input/2.bmp');
% I = im2double(I);
% PSF = fspecial ('gaussian', 10, 5);     
% J = imfilter(I, PSF, 'conv');           %图像退化
% v = 0.02;
% K = imnoise(J, 'gaussian', 0, v);       %添加噪声
% NP = v*prod(size(I));
% [L, LAGRA] = deconvreg(K, PSF, NP);     %图像复原
% edged = edgetaper(K, PSF);              %提取边缘
% figure,
% subplot(131), imshow(I);
% subplot(132), imshow(K);
% subplot(133), imshow(edged);
% M1 = deconvreg(edged, PSF, [], LAGRA);   %图像复原
% M2 = deconvreg(edged, PSF, [], LAGRA*30);%增大拉格朗日算子
% M3 = deconvreg(edged, PSF, [], LAGRA/60);%减小拉格朗日算子
% figure,
% subplot(131), imshow(M1);
% subplot(132), imshow(M2);
% subplot(133), imshow(M3);
%% Lucy-Richardson算法对图像进行复原
% J = deconvlucy(I, PSF, NUMIT, DAMPAR, WEIGHT, READOUT, SUBSMPL)
% NUMIT 为算法重复次数，默认值为10，DAMPAR为偏差阈值，默认值为0，
% WEIGHT 为像素加权值，默认为原始图像的数值。
% READOUT 为噪声矩阵，默认值为0
% SUBSMPL 为子采样时间，默认值为1
% I = imread('input/2.bmp');
% I = im2double(I);
% LEN = 30;
% THETA = 20;
% PSF = fspecial('motion', LEN, THETA);
% J = imfilter(I, PSF, 'circular', 'conv');
% figure,
% subplot(121), imshow(I);
% subplot(122), imshow(J);
% K = deconvlucy(J, PSF, 5);
% L = deconvlucy(J, PSF, 15);
% figure,
% subplot(121), imshow(K);
% subplot(122), imshow(L);
%% 高斯噪声采用Lucy-Richardson算法进行图像复原
% I = imread ('input/2.bmp');
% I = im2double(I);
% PSF = fspecial('gaussian', 7, 10);
% v = 0.0001;
% J = imnoise (imfilter(I, PSF), 'gaussian', 0, v);   %图像退化
% figure,
% subplot(121), imshow(I);
% subplot(122), imshow(J);
% WT = zeros(size(I));
% WT (5:end-4, 5:end-4) = 1;
% K = deconvlucy(J, PSF, 20, sqrt(v));                %图像复原
% L = deconvlucy(J, PSF, 20, sqrt(v), WT);
% figure,
% subplot(121), imshow(K);
% subplot(122), imshow(L);
%% 盲解卷积复原
% [J, PSF] = deconvblind(I, INITPSF, NUMIT, DAMPAR, WEIGHT, READOUT)
% INITPSF 为PSF估计值，返回值为实际采用的PSF值,NUMIT为算法重复次数默认为10
% DAMPAR为偏移阈值，默认值为0，WEIGHT为像素加权值，默认为原始图像值。
% READOUT 为噪声矩阵
% I = imread ('input/2.bmp');
%I = rgb2gray(I);
% I = im2double (I);
% LEN = 20;
% THETA = 20;
% PSF = fspecial('motion', LEN, THETA);
% J = imfilter(I, PSF, 'circular', 'conv');   %运动模糊
% INITPSF = ones(size(PSF));
% [K, PSF2] = deconvblind(J, INITPSF, 30);     %图像复原
% figure,
% subplot(121), imshow (PSF, []);
% subplot(122), imshow (PSF2, []);
% axis auto;
% figure,
% subplot(121), imshow (J);
% subplot(122), imshow (K);
%% 对退化图像进行盲卷积复原
% checkerborad 函数：创建棋盘图像
% checkerboard 创建一个8*8个单元的棋盘图像，每个单元正方形，边长为10个像素.
% 亮的部分为白色，暗的部分为黑色
% I = checkerboard(n, p, q) 创建一个2p*2q个单元的棋盘图像，每个单元边长为n个像素
I = checkerboard(8);                  %产生图像
PSF = fspecial('gaussian', 7, 10);  %建立PSF
v = 0.001;
J = imnoise(imfilter(I, PSF), 'gaussian', 0, v);    %图像退化
INITPSF = ones(size(PSF));
WT = zeros(size(I));
WT(5:end-4, 5:end-4) = 1;
[K, PSF2] = deconvblind(J, INITPSF, 20, 10*sqrt(v), WT); %图像复原
figure,
subplot(131), imshow (I);
subplot(132), imshow (J);
subplot(133), imshow (K);