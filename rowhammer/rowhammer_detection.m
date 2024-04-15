clear all;
clc

%% load file
Fs = 25e6;
% file_name = "./rowhammer/ddr4_1066_20s_single_sided.sc16q11";
file_name = 'rowhammer/ddr4_1066_40s_single_side_clflush_2.sc16q11';
% file_name = 'rowhammer/ddr4_1066_40s_daily_use_1.sc16q11';
s_dat = load_sc16q11(file_name);

t =40;
s = s_dat(1:Fs*t);

%% FFT

L =length(s);

Y1 = fft(s)/L;

Sf = fftshift(abs(fft(s))) /L;

fft_size = 8192;

avg_count =20; 

num_sample = 1000;

num_spectrm = 12;

movement = fix(fft_size/2);



period = Fs*0.04;

labels = zeros(num_sample,1);

for i = 1:num_sample
    
    start_point = (i-1)*(period);
    ffts = zeros(num_spectrm,fft_size);
    for t = 1:num_spectrm
        avg_ffts = zeros(avg_count,fft_size);
        for n = 1:avg_count
            s_temp = s(start_point+((n - 1)*movement)+1:start_point+((n - 1)*movement)+fft_size);
            fft_temp = fftshift(abs(fft(s_temp)))/fft_size;
            avg_ffts(n,1:end) = fft_temp(1:end);
        end
        temp = mean(avg_ffts);
        ffts(t,1:end) = temp(1:end);
        ffts_con = ffts';
        if (mean(ffts_con(840))) >= 0.022
             labels(i) =1;
        end
        
    end

    % normalization
    ffts_nor = ffts/max(max(ffts));
    writematrix(ffts_nor,'True.csv','WriteMode','append');
    
    % 输出二维矩阵
end
writematrix(labels,'labels.csv','WriteMode','append');

figure(1)
plot (ffts');
%% 1000*12*8192
