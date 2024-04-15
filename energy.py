import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.pyplot import MultipleLocator
from scipy.fft import fft, fftfreq, fftshift
import os
import sys
import io
import cv2
import h5py
def convert_plaintext_key(file_path, key_num):
    """try:
        pks = np.loadtxt(file_path, dtype=np.int16, delimiter = ',')
    except:
        print("Error2: can't open txt file for read")
        sys.exit(-1)
    print(pks.shape)
    plaintext = np.zeros((key_num, 16), dtype=np.int16)
    aes_key = np.zeros((key_num, 16), dtype=np.int16)
    for n in range(int(len(pks)/2)):
        plaintext[n] = pks[2*n+1]
        aes_key[n] = pks[2*n]
    return plaintext, aes_key"""
    plaintext = np.zeros((key_num, 16), dtype=int)
    aes_key = np.zeros((key_num, 16), dtype=int)
    aes_runtime = np.zeros(key_num+1, dtype=int)
    total_runtime = np.zeros(key_num+1, dtype=int)
    with open(file_path, 'r') as file:
        lines = file.readlines();
        count =0
        for line in lines:
            if count%4 == 0:
                aes_runtime[count // 4] = int(line)
            elif count%4 == 1:
                total_runtime[count // 4] = int(line)
            elif count%4 == 2:
                plaintext[count // 4] = np.array([int(s) for s in line.split(',')])
            elif count%4 == 3:
                aes_key[count // 4] = np.array([int(s) for s in line.split(',')])
            count+=1
    return aes_runtime, total_runtime, aes_key, plaintext

def read_and_fft(file, points_to_read, num_to_seek):
    file.seek((num_to_seek) * points_to_read)
    bytes = file.read(points_to_read)
    memArray = np.frombuffer(bytes, dtype='<f4').copy()
    trace = memArray[0::2] + 1j * memArray[1::2]
    fft_energy = fft(trace)
    x = fftfreq(points_to_read//8, d=1/80000000)

    plt.figure()
    plt.plot(x, abs(fft_energy))
    plt.show()
if __name__ == "__main__":
    sample_rate = 10000000
    key_num = 100
    sample_per_key = 100
    file_path = "aes_100_record.raw"
    key_file_path = "aes_100_record.txt"
    aes_runtime, execTime, aes_key, plaintext = convert_plaintext_key(key_file_path,key_num)
    window_size = 2048
    #points_per_sample = 4000
    points_to_read = window_size * 8
    #seek_num = (points_per_sample - window_size) // sample_per_key
    #print(seek_num)
    ###
    with open(file_path, 'rb') as file:
        sample_energy = np.zeros((key_num,sample_per_key,window_size//2))
        for i in range (key_num):
            seek_execPoint = (np.sum(execTime[:i+2]) - aes_runtime[i+1]) * 10
            points_per_sample = aes_runtime[i+1]*10
            seek_num = (points_per_sample - window_size) // sample_per_key
            for n in range(sample_per_key):
                file.seek((seek_execPoint + (n * seek_num))*8)
                bytes = file.read(points_to_read)
                memArray = np.frombuffer(bytes, dtype='<f4').copy()

                memArray = memArray[0::2] + 1j * memArray[1::2]
                memArray -= np.mean(memArray)
                fft_energy = fft(memArray)
                fft_size = int(len(fft_energy)/2)
                """energy = np.zeros(fft_size+1)
                energy[0] = np.abs(fft_energy[0]) / len(fft_energy)
                energy[fft_size] = np.abs(fft_energy[fft_size]) / len(fft_energy)
                energy[1:fft_size] = np.abs(fft_energy[1:fft_size]) * 2 / len(fft_energy)"""
                # energy_sum = np.sum(energy)
                # energy /= np.max(energy)
                sample_energy[i][n] = fft_energy[:fft_size]
                #sample_energy[i][n] = energy_sum
            #sample_energy[t]/=np.max(sample_energy[t])
        #sample_energy = sample_energy / np.max(sample_energy)
        print(sample_energy.shape)
        samples = np.linspace(0, sample_per_key * window_size//2, sample_per_key * window_size//2)
        samples.shape = (sample_per_key * window_size//2,1)
        #samples = np.linspace(0, (sample_per_key*key_num), (sample_per_key*key_num))
        #samples.shape = ((key_num*sample_per_key),1)

        ax = plt.gca()
        #ax.yaxis.set_major_locator(MultipleLocator(np.max(sample_energy) / 0.2))
        #ax.xaxis.set_major_locator(MultipleLocator(400 / 50))
        plt.figure(figsize=(19.2,10.8))
        #plt.ylim(0,2)
        #sample_energy_full = np.zeros((key_num*sample_per_key), dtype='f4')
        totalsample = np.zeros(sample_per_key*window_size*10//2)
        samples_10 = np.linspace(0, sample_per_key * window_size*10//2, sample_per_key * window_size*10//2)
        samples_10.shape = (sample_per_key * window_size*10//2,1)
        for i in range (10):
            tempsample = np.zeros(sample_per_key*window_size//2)
            for n in range(sample_per_key):
                totalsample[(i*sample_per_key*window_size//2 + n*window_size//2): (i*sample_per_key*window_size//2 + (n+1)*window_size//2)] = sample_energy[i][n][0:window_size//2]
                tempsample[(n*window_size//2):((n+1)*window_size//2)] = sample_energy[i][n][0:window_size//2]
            # plt.plot(samples, tempsample)
        plt.plot(samples_10,totalsample)
        #sample_energy_full.shape = ((key_num*sample_per_key),1)
        #plt.plot(samples, sample_energy_full)
        plt.savefig('aes_100_total_time_plot.jpg')
        plt.show()
        """points_to_read = int(time_record * sample_rate / (real_key_num * 1000 * sample_per_key)) * 8
        for i in [0, 100, 500, 1000, 2000, 4000]:
            read_and_fft(file, points_to_read, i)"""