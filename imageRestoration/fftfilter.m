function Z = fftfilter(X, H)
F = fft2(X, size(H, 1), size(H, 2));       %傅里叶变换
Z = H.*F;                                   %滤波
Z = ifftshift(Z);
Z = abs(ifft2(Z));                          %傅里叶反变换
Z = Z(1:size(X, 1), 1:size(X, 2));
end