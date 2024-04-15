clear all; clc
%% Define the parameters for the clock signal and modulation

% Define parameters
A = 1;        % Amplitude
f0 = 1e6;     % unspread clock frequency
fm = 2e4;     % Modulation frequency
df = 1.8e4;   % Peak frequency deviation
T = 1/f0;     % Period
fs = 10e6;    % Sampling rate
t = 0:1/fs:10000*T;   % Time vector

% Define unspread clock
Vclk = A*cos(2*pi*f0*t);
% Define spread spectrum clock
Vssc = A*cos(2*pi*f0*t + df/fm*sin(2*pi*fm*t));

% harmonic model
% nn = 3;
% Vssc = A*cos(2*pi*nn*f0*t + df/fm*sin(2*pi*fm*t));

%% plot
% Plot the signals
% figure;
% plot(t, Vclk, 'b');
% hold on;
% plot(t, Vssc, 'r');
% xlabel('Time (s)');
% ylabel('Amplitude (V)');
% legend('Unspread clock', 'Spread spectrum clock');
% title('Time domain signals');

% Plot the frequency spectrum of unspread clock and spread spectrum clock
L = length(Vclk);
NFFT = 2 * 2^nextpow2(L);
f = fs/2*linspace(0,1,floor(L/2)+1);
Y1 = fft(Vclk)/L;
Y2 = fft(Vssc)/L;

figure (1);
subplot(2,1,1)
plot(f, 2*abs(Y1(1:floor(L/2)+1)),'b');   % 0:fs/2, real freq
set(gca,'XLim',[0.9e6  1.1e6]);
set(gca,'YLim',[0      1.5]);
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Frequency spectrum of unspread clock');

subplot(2,1,2)
plot(f, 2*abs(Y2(1:floor(L/2)+1)),'r');
set(gca,'XLim',[0.9e6  1.1e6]);
set(gca,'YLim',[0      1.5]);
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Frequency spectrum of spread spectrum clock');

%% phrase
Vssc_hilbert = hilbert(Vssc);              % Compute the Hilbert transform of the SSC signal 改变信号相位
inst_phase = unwrap(angle(Vssc_hilbert));  % Compute the instantaneous phase 得到相位角序列
diff_phase = diff(inst_phase); % 得到相位差序列

figure (2);
plot(diff_phase);
set(gca,'XLim',[1000  2500]);
set(gca,'YLim',[0.55  0.7]);
xlabel('Sample');
ylabel('Delta Phase');
title('Phase difference between successive sampels');
%相位差是正弦周期

%% De-spread the SSC signal
% 1 Period: 500 samples, max: 0.639653 , min: 0.616986
% y = A*sin(2πft + φ) + D
st = 1:1:100001;
% theater = 0.0113 * sin(2 * pi * 0.002*st+1.5) + 0.6283;
theater = 0.0113 * sin(2 * pi * 0.002*st+1.55) + 0;

figure (3);
plot(theater);
set(gca,'XLim',[1000  2500]);
% set(gca,'YLim',[0.55  0.7]);
title('Theater');

phase_cumsum = cumsum(theater); %累计和
Vssc_recoverd= Vssc_hilbert .* exp(-1i*phase_cumsum); %对矩阵每个元素进行指数运算
Y3 = fft(Vssc_hilbert, 10e6)/L;
Y4 = fft(Vssc_recoverd,10e6)/L;

figure (4);
subplot(2,1,1)
plot(abs(Y3),'b');
% set(gca,'XLim',[0      0.02e4]);
set(gca,'YLim',[0  1.5]);
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Frequency spectrum of Vssc hilbert');

subplot(2,1,2)
plot(abs(Y4),'r');
% plot(f, 2*abs(Y3(1:floor(L/2)+1)),'r');
% set(gca,'XLim',[0      0.02e4]);
set(gca,'YLim',[0  1.5]);
xlabel('Frequency (Hz)');
ylabel('Amplitude');
title('Frequency spectrum of non-SSC');

%% phrase
% cutoff_freq = 10e4; % Cutoff frequency (Hz)
% filter_order= 100;  % Order of the filter
% lp_filter = designfilt('lowpassfir', 'FilterOrder', filter_order, ...
%                        'CutoffFrequency', cutoff_freq, 'SampleRate', fs);
% % Apply the lowpass filter to the complex signal
% filtered_signal = filtfilt(lp_filter, Vssc);
% phase = angle(filtered_signal);
% phase_diff = diff(phase);       % Calculate the phase change (difference)
% phase_diff_unwrapped = unwrap(phase_diff); % Unwrap the phase difference
% inst_freq = diff(inst_phase) / (2 * pi * 1/fs); % Compute the instantaneous frequency
% mod_signal = sin(2 * pi * fm * t);  % Generate the sinusoidal modulation signal
% inst_freq_no_mod = inst_freq - 0.01 * f0 * mod_signal(1:end-1); % Remove the modulation
% orig_phase = cumsum(inst_freq_no_mod) * (2 * pi * 1/fs); % Integrate the de-modulated instantaneous frequency
% ss_clock_recov = sin(orig_phase);              % Generate the recovered clock signal
% f_ss_clock_recov= abs(fft(ss_clock_recov));