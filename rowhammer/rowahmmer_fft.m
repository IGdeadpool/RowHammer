clear all;
clc

file_name = 'rowhammer/ddr4_1066_40s_single_side_clflush_2.sc16q11';

s_dat = load_sc16q11(file_name);
Fs = 25e6;
t =40;
s = s_dat(1:Fs*t);

L =length(s);

Y1 = fft(s)/L;

Sf = fftshift(abs(fft(s))) /L;

figure(1)
plot(Sf)