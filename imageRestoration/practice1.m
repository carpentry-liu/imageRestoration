close all; clear all; clc;
%% ͨ��λ����Ϣ��������˹����
% I = imread ('input/2.bmp');
% I = im2double(I);
% V = zeros(size(I));                 %��������V
% for i = 1:size(V, 1)
%     V(i, :) = 0.02*i/size(V, 1);
% end
% J = imnoise(I, 'localvar', V);      %��Ӹ�˹����
% figure,
% subplot(121), imshow(I);
% subplot(122), imshow(J); 
% %% ͨ������ֵ��������˹����
% I = imread ('input/1.jpg');
% I = im2double(I);
% h = 0:0.1:1;
% v = 0.01:-0.001:0;
% J = imnoise (I, 'localvar', h, v);      %�������
% figure,
% subplot(121), imshow(I);
% subplot(122), imshow(J);
%% ��ӽ�������
% % �����ܶ�Ϊd�� ������ռ�������������İٷֱȣ�ϵͳĬ��0.05
% % salt & pepper , ����&ǰ������пո�
% % �������Ժڵ㣨�����㣩������ͼ����
% % �������԰׵㣨�ε㣩������ͼ����
% % ȥ�����������ȽϺõķ�������ֵ�˲�
% I = imread ('input/1.jpg');
% I = im2double(I);
% %J = imnoise(I, 'salt & pepper', 0.01);     %��ӽ�������
% %K = imnoise(I, 'salt & pepper', 0.03);
% % R = rand(size(I));
% % J = I;
% % J (R<=0.02) = 0;                            %��ӽ�����
% % K = I;
% % K (R<=0.03) = 1;                            %���������
% J = imnoise (I, 'poisson');                   %��Ӳ�������
% figure,
% subplot(121), imshow (I);
% subplot(122), imshow (J);
%% ��ӳ������� J = imnoise (I, 'speckle', v)
% % J = I * n * I , n Ϊ��ֵΪ0������Ϊv�ľ��ȷֲ����������, Ĭ��ֵ0.04
% I = imread ('input/2.bmp');
% J = imnoise (I, 'speckle');             %��ӳ�������
% K = imnoise (I, 'speckle', 0.2);
% figure,
% subplot(121), imshow (J);
% subplot(122), imshow (K);
% %% �������ȷֲ�����
% m = 256; n = 256;
% a = 50; b = 180;
% I = a + (b  - a) * rand (m, n);         %���ȷֲ�����
% figure,
% subplot(121), imshow(uint8(I));         %��ʾ����ͼ��
% subplot(122), imhist(uint8(I));
% %% ����ָ���ֲ�����
% m = 256; n = 256;
% a = 0.04;
% K = -1/a;
% I = K*log(1-rand(m,n));                 %ָ���ֲ�����
% figure,
% subplot(121), imshow(uint8(I));         %��ʾ����ͼ��
% subplot(122), imhist(uint8(I));         %��ʾֱ��ͼ
%% ��ͼ�����������ֵ�ͼ��ξ�ֵ�˲�
% % fspecial�������ڽ���Ԥ������˲�����
% % h = fspecial(type) h = fspecial(type, para)
% I = imread ('input/2.bmp');
% I = im2double(I);
% I = imnoise (I, 'gaussian', 0, 0.05);       %��ֵΪ0������Ϊ0.05�ĸ�˹��������ֵĬ��ֵΪ0������ 0.01
% PSF = fspecial ('average', 3);              %����PSF
% J = imfilter(I, PSF);                       %������ֵ�˲�
% K = exp(imfilter(log(I), PSF));             %���ξ�ֵ�˲�
% figure,
% subplot(131), imshow(I);                    %��ʾ��������ͼ��
% subplot(132), imshow(J);
% subplot(133), imshow(K);
%% ��г����ֵ�˲���
% I = imread ('input/2.bmp');
% I = im2double(I);
% I = imnoise (I, 'salt & pepper', 0.01);     %��ӽ�������
% PSF = fspecial ('average', 3);              %����PSF
% Q1 = 1.6; Q2 = -1.6;
% j1 = imfilter(I.^(Q1+1), PSF);
% j2 = imfilter(I.^Q1, PSF);
% J = j1./j2;                                  %��г�˲���QΪ����ȥ��������(�ڵ㣩
% K1 = imfilter(I.^(Q2+1), PSF);
% K2 = imfilter(I.^Q2, PSF);
% K = K1./K2;                                  %��г�˲���QΪ����ȥ�����������׵㣩
% figure,
% subplot(131), imshow(I);
% subplot(132); imshow(J);
% subplot(133); imshow(K);
%% ˳��ͳ���˲�����ֵ�˲��ܺܺñ���ͼ���Ե���ʺ�ȥ������������Ч�����ھ�ֵ�˲���
% % ���ֵ�˲�Ҳ��ȥ����������������Ӻ�ɫ��Եȥ��һЩ��ɫ�ء�
% % ͬ��  ��Сֵ�˲�����Ӱ�ɫ�����Եȥ��һЩ��ɫ�ء�
% % J = ordfilt2(I, order, domain):
% %��ͼ����ж�ά�����˲�������domain�з�0ֵ��������orderΪѡ������λ��
% I = imread ('input/2.bmp');
% I = rgb2gray(I);
% I = im2double(I);
% I = imnoise (I, 'salt & pepper', 0.05);         %��ӽ��������� ϵͳĬ��0.05
% J = medfilt2(I, [3, 3]);                        %��ά��ֵ�˲�
% domain = [0 1 1 0; 1 1 1 1; 1 1 1 1; 0 1 1 0];     %����ģ��
%J = ordfilt2(I, 6, domain);        
%J = ordfilt2(I, 1, ones(4, 4));                   %���ֵ�˲�
%J = ordfilt2(I, 9, ones(3));                      %��Сֵ�˲�
% figure;
% subplot (121);imshow (I);
% subplot (122);imshow (J);
%% ����Ӧά���˲�
% % J = wiener2(I, [m, n], noise):���ô���m*n Ĭ��3*3��noiseΪ����������
% % [J, noise] = wiener2(I, [m, n]):  ���������й��ƣ�noiseΪ����������
% % imcrop(I, [a, b, c, d]);  (a,b)��ʾ�ü������Ͻ�������ԭͼ���е�λ�ã�
% % c��ʾ�ü���Ŀ�d��ʾ�ü���ĸ�
% I = imread ('input/2.bmp');
% I = rgb2gray(I);
% I = imcrop(I, [100, 100, 1024, 1024]);          %ͼ�����
% J = imnoise(I, 'gaussian', 0, 0.03);               %�������
% [K, noise] = wiener2(J, [5, 5]);                %����Ӧ�˲�
% figure,
% subplot(121), imshow(J), axis on;
% subplot(122), imshow(K), axis on;
%% ���˲�����ͼ����и�ԭ
% [X, Y] = meshgrid(x, y) 
% ���X��ÿһ�е���ֵ���Ǹ��Ƶ�x��ֵ�����Y��ÿһ�ж�����ֵ���Ǹ���y��ֵ
% imshow(I, []) �൱��imshow[min(A(:)),max(A(:))],ʹ����Сֵ��Ϊ��ɫ�����ֵ��Ϊ��ɫ��
% I = imread ('input/1.jpg');
% I = im2double(I);
% [m, n] = size (I);
% M = 2*m; N = 2*n;
% u = -m/2:m/2-1;
% v = -n/2:n/2-1;
% [U, V] = meshgrid(u, v);
% D = sqrt (U.^2 + V.^2);
% D0 = 130;                          %��ֹƵ��
% H = exp(-(D.^2)./(2*(D0^2)));       %��˹��ͨ�˲���
% N = 0.01*ones(size(I, 1), size(I, 2));
% N = imnoise(N, 'gaussian', 0, 0.001);%�������
% J = fftfilter(I, H) + N;              %Ƶ���˲�����������
% figure,
% subplot(121), imshow(I);               %��ʾԭʼͼ��
% subplot(122), imshow(J, []);            %��ʾ�˻����ͼ��
% HC = zeros(m, n);
% M1 = H > 0.1;                           %Ƶ�ʷ�Χ
% HC(M1) = 1./H(M1);
% K = fftfilter(J, HC);                   %���˲�
% HC = zeros(m, n);
% M2 = H > 0.01;
% HC(M2) = 1./H(M2);
% L = fftfilter(J, HC);                   %�������˲�
% figure,
% subplot (121), imshow (K, []);          %��ʾ���
% subplot (122), imshow (L, []);
%% ά���˲���ԭ
% J = deconvwnr(I, PSF, NSR) PSFΪ����չ������NSRΪ�����
% J = deconvwnr(I, PSF, NCORR, ICORR) NCORRΪ����������غ�����ICORRΪԭʼͼ������غ���
% I = imread('input/1.jpg');
% I = rgb2gray(I);
% I = im2double(I);
% LEN = 25;                   %��������
% THETA = 20;
% PSF = fspecial('motion', LEN, THETA);   %����PSF���˶�λ��Ϊ25�����أ��Ƕ�Ϊ20��
% J = imfilter(I, PSF, 'conv', 'circular');%�˶�ģ��
% NSR = 0;
% K = deconvwnr(J, PSF, NSR);             %ά���˲���ԭ
% figure,
% subplot(131), imshow(I);
% subplot(132), imshow(J);
% subplot(133), imshow(K);
%% ά���˲��Ժ����������˶�ģ��ͼ����и�ԭ
% I = imread ('input/1.jpg');
% I = rgb2gray(I);
% I = im2double(I);
% LEN = 21;
% THETA = 11;
% PSF = fspecial('motion', LEN, THETA);
% J = imfilter(I, PSF, 'conv', 'circular');           %�����˶�ģ��
% noise_mean = 0;
% noise_var = 0.0001;
% K = imnoise(J, 'gaussian', noise_mean, noise_var)  %��Ӹ�˹����
% figure,
% subplot(121), imshow(I);
% subplot(122), imshow(K);
% NSR1 = 0;
% L1 = deconvwnr(K, PSF, NSR1);                   %ά���˲���ԭ
% NSR2 = noise_var/var(I(:));
% L2 = deconvwnr(K, PSF, NSR2);           
% figure,
% subplot(121), imshow(L1);
% subplot(122), imshow(L2);
%% �������Ϣ���и�ԭ
% I = imread('input/2.bmp');
% I = im2double(I);
% LEN = 20;                                       %���ò���
% THETA = 10;
% PSF = fspecial('motion', LEN, THETA);           %����PSF
% J = imfilter(I, PSF, 'conv', 'circular');        %�˶�ģ��
% figure,
% subplot(121), imshow(I);
% subplot(122), imshow(J); 
% noise = 0.03*randn(size(I));
% K = imadd(J, noise);                        %�������
% NP = abs(fft2(noise)).^2;                   
% NPower = sum(NP(:))/prod(size(noise));
% NCORR = fftshift(real(ifft2(NP)));          %����������غ���
% IP = abs (fft2(I)).^2;
% IPower = sum(IP(:))/prod(size(I));
% ICORR = fftshift(real(ifft2(IP)));          %ͼ�������غ���
% L = deconvwnr(K, PSF, NCORR, ICORR);        %ͼ��ԭ
% figure,
% subplot(121), imshow(K);                    %��ʾ���
% subplot(122), imshow(L);                    %��ʾ���
%% Լ����С���˷���ԭ
% J = deconvreg(I, PSF, NOISEPOWER�� LRANGE, REGOP) 
% PSF Ϊ����չ���� NOISEPOWERΪ����ǿ�ȣ�Ĭ��ֵΪ0 LRANGEΪ������������������Χ��Ĭ��ֵΪ[10^-9, 10^9]
% REGOP ΪԼ������
% [J, LAGRA] = deconvreg(I, PSF,...)  ����ֵLAGREΪ���ղ��õ�������������
% I = imread ('input/2.bmp');
% I = im2double(I);
% PSF = fspecial('gaussian', 8, 4);       %����PSF
% J = imfilter(I, PSF, 'conv');           %ͼ���˻�
% figure,
% subplot(121), imshow(I);
% subplot(122), imshow(J);                %��ʾ�˻����ͼ��
% v = 0.02;
% K = imnoise (J, 'gaussian', 0, v);      %�������
% NP = v*prod(size(I));
% L = deconvreg(K, PSF, NP);              %ͼ��ԭ
% figure,
% subplot(121), imshow(K);
% subplot(122), imshow(L);
%% �����������ӽ���ͼ��ԭ
% I = imread('input/2.bmp');
% I = im2double(I);
% PSF = fspecial ('gaussian', 10, 5);     
% J = imfilter(I, PSF, 'conv');           %ͼ���˻�
% v = 0.02;
% K = imnoise(J, 'gaussian', 0, v);       %�������
% NP = v*prod(size(I));
% [L, LAGRA] = deconvreg(K, PSF, NP);     %ͼ��ԭ
% edged = edgetaper(K, PSF);              %��ȡ��Ե
% figure,
% subplot(131), imshow(I);
% subplot(132), imshow(K);
% subplot(133), imshow(edged);
% M1 = deconvreg(edged, PSF, [], LAGRA);   %ͼ��ԭ
% M2 = deconvreg(edged, PSF, [], LAGRA*30);%����������������
% M3 = deconvreg(edged, PSF, [], LAGRA/60);%��С������������
% figure,
% subplot(131), imshow(M1);
% subplot(132), imshow(M2);
% subplot(133), imshow(M3);
%% Lucy-Richardson�㷨��ͼ����и�ԭ
% J = deconvlucy(I, PSF, NUMIT, DAMPAR, WEIGHT, READOUT, SUBSMPL)
% NUMIT Ϊ�㷨�ظ�������Ĭ��ֵΪ10��DAMPARΪƫ����ֵ��Ĭ��ֵΪ0��
% WEIGHT Ϊ���ؼ�Ȩֵ��Ĭ��Ϊԭʼͼ�����ֵ��
% READOUT Ϊ��������Ĭ��ֵΪ0
% SUBSMPL Ϊ�Ӳ���ʱ�䣬Ĭ��ֵΪ1
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
%% ��˹��������Lucy-Richardson�㷨����ͼ��ԭ
% I = imread ('input/2.bmp');
% I = im2double(I);
% PSF = fspecial('gaussian', 7, 10);
% v = 0.0001;
% J = imnoise (imfilter(I, PSF), 'gaussian', 0, v);   %ͼ���˻�
% figure,
% subplot(121), imshow(I);
% subplot(122), imshow(J);
% WT = zeros(size(I));
% WT (5:end-4, 5:end-4) = 1;
% K = deconvlucy(J, PSF, 20, sqrt(v));                %ͼ��ԭ
% L = deconvlucy(J, PSF, 20, sqrt(v), WT);
% figure,
% subplot(121), imshow(K);
% subplot(122), imshow(L);
%% ä������ԭ
% [J, PSF] = deconvblind(I, INITPSF, NUMIT, DAMPAR, WEIGHT, READOUT)
% INITPSF ΪPSF����ֵ������ֵΪʵ�ʲ��õ�PSFֵ,NUMITΪ�㷨�ظ�����Ĭ��Ϊ10
% DAMPARΪƫ����ֵ��Ĭ��ֵΪ0��WEIGHTΪ���ؼ�Ȩֵ��Ĭ��Ϊԭʼͼ��ֵ��
% READOUT Ϊ��������
% I = imread ('input/2.bmp');
%I = rgb2gray(I);
% I = im2double (I);
% LEN = 20;
% THETA = 20;
% PSF = fspecial('motion', LEN, THETA);
% J = imfilter(I, PSF, 'circular', 'conv');   %�˶�ģ��
% INITPSF = ones(size(PSF));
% [K, PSF2] = deconvblind(J, INITPSF, 30);     %ͼ��ԭ
% figure,
% subplot(121), imshow (PSF, []);
% subplot(122), imshow (PSF2, []);
% axis auto;
% figure,
% subplot(121), imshow (J);
% subplot(122), imshow (K);
%% ���˻�ͼ�����ä�����ԭ
% checkerborad ��������������ͼ��
% checkerboard ����һ��8*8����Ԫ������ͼ��ÿ����Ԫ�����Σ��߳�Ϊ10������.
% ���Ĳ���Ϊ��ɫ�����Ĳ���Ϊ��ɫ
% I = checkerboard(n, p, q) ����һ��2p*2q����Ԫ������ͼ��ÿ����Ԫ�߳�Ϊn������
I = checkerboard(8);                  %����ͼ��
PSF = fspecial('gaussian', 7, 10);  %����PSF
v = 0.001;
J = imnoise(imfilter(I, PSF), 'gaussian', 0, v);    %ͼ���˻�
INITPSF = ones(size(PSF));
WT = zeros(size(I));
WT(5:end-4, 5:end-4) = 1;
[K, PSF2] = deconvblind(J, INITPSF, 20, 10*sqrt(v), WT); %ͼ��ԭ
figure,
subplot(131), imshow (I);
subplot(132), imshow (J);
subplot(133), imshow (K);