import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    t = 0.01
    window_size = int(1000*1000*10 * t)
    sample_freq = 10000000
    total_t = 1
    total_point = total_t * sample_freq


    resolution = sample_freq/window_size
    file_name = 'uno_80MHZ_empty_10s.sc16q11'

    fft_count = 2000

    plt.figure(figsize=(20.0, 8.0))
    energy_collection = np.zeros(fft_count)
    with open(file_name, 'r') as file:
        points_to_read = window_size*8
        movement = ((total_point-window_size)//fft_count)*8
        for i in range(fft_count):
            file.seek(i*movement)
            bytes = file.read(points_to_read)
            memArray = np.frombuffer(bytes, dtype='<f4').copy()
            signal = memArray[0::2] + 1j * memArray[1::2]
            fft_y = fft(signal)

            fft_no_abs = np.array(fft_y / window_size * 2)
            fft_no_abs[0] = 0.5 * fft_no_abs[0]
            fft_no_abs_shift = fftshift(fft_no_abs)
            required_length = 100
            range_middle = int(required_length / resolution)

            fft_no_abs_shift_middle = fft_no_abs_shift[window_size//2 - range_middle:window_size//2 + range_middle]
            energy_sum = np.sum(np.abs(fft_no_abs_shift_middle))

            energy_collection[i] = energy_sum



        fft_energy = fft(energy_collection)
        fft_energy_shift = fftshift(np.array(fft_energy / len(fft_energy)*2))
        list0 = np.linspace(0,fft_count, fft_count)
        list0.shape = (fft_count,1)


        list_100 = np.linspace(0,300,300)
        list_100.shape = (300,1)
        plt.subplot(211)
        plt.plot(list0, energy_collection)
        plt.subplot(212)
        plt.plot(list_100, energy_collection[800:1100])
        file.flush()
        file.close()

    plt.show()
