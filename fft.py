
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import os
import sys
import io
import cv2
import h5py
import keyboard


AES_Sbox = np.array([
			0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
			0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
			0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
			0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
			0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
			0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
			0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
			0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
			0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
			0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
			0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
			0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
			0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
			0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
			0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
			0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
			])
# function to convert recorded AES key
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
    aes_runtime = np.zeros(key_num + 1, dtype=int)
    total_runtime = np.zeros(key_num + 1, dtype=int)
    with open(file_path, 'r') as file:
        lines = file.readlines();
        count = 0
        for line in lines:
            if count % 4 == 0:
                aes_runtime[count // 4] = int(line) + 4
            elif count % 4 == 1:
                total_runtime[count // 4] = int(line)
            elif count % 4 == 2:
                plaintext[count // 4] = np.array([int(s) for s in line.split(',')])
            elif count % 4 == 3:
                aes_key[count // 4] = np.array([int(s) for s in line.split(',')])
            count += 1
    return aes_runtime, total_runtime, aes_key, plaintext

#convert .raw to complex
def convert_raw_to_dataset(traces_file_path, train_group, valid_group,  key_num, sample_per_key, aes_runtime, execTime, window_size, validaion_rate):

    train_traces = np.zeros((int(key_num*(1.0-validaion_rate)), sample_per_key), dtype='f4')
    validation_traces = np.zeros((int(key_num*validaion_rate), sample_per_key), dtype='f4')
    points_to_read = window_size*8
    #train_traces = np.zeros((int(key_num*(1.0-validaion_rate))*16,((length_signal * sample_per_key)//16)), dtype='f4')
    #validation_traces = np.zeros((int(key_num*validaion_rate)*16,((length_signal * sample_per_key)//16)), dtype='f4')
    with open(traces_file_path, 'rb') as file:
        for i in range(int(key_num*(1.0-validaion_rate))):
            seek_execPoint = (np.sum(execTime[:i + 2]) - aes_runtime[i]) * 10 * 8
            points_per_sample = aes_runtime[i] * 10
            seek_num = (points_per_sample - window_size) // sample_per_key
            for n in range(sample_per_key):
                file.seek(seek_execPoint + (n * seek_num * 8))
                bytes = file.read(points_to_read)
                memArray = np.frombuffer(bytes, dtype='<f4').copy()
                trace = memArray[0::2] + 1j * memArray[1::2]
                fft_energy = fft(trace)
                energy = np.square(np.abs(fft_energy).astype('f4'))
                energy_sum = np.sum(energy)
                """#trace_point = i * sample_per_key +n
                #file.seek(trace_point * points_to_read)
                bytes = file.read(points_to_read)
                memArray = np.frombuffer(bytes, dtype='<f4').copy()
                trace = memArray[0::2] + 1j * memArray[1::2]
                fft_energy = fft(trace)
                energy = np.square(np.abs(fft_energy))
                energy_sum = np.sum(energy)"""
                train_traces[i][n] = energy_sum
            # train_traces[i] /= np.max(train_traces[i])
            print('trace :', i)    #train_traces[i*16+n] = energy
        # collect data of validation set
        ###
        print("processing valid trace")
        ###
        for i in range(int(key_num*(1.0-validaion_rate)), key_num):
            seek_execPoint = (np.sum(execTime[:i + 2]) - aes_runtime[i]) * 10 * 8
            points_per_sample = aes_runtime[i] * 10
            seek_num = (points_per_sample - window_size) // sample_per_key
            for n in range(sample_per_key):
                file.seek(seek_execPoint + (n * seek_num * 8))
                bytes = file.read(points_to_read)
                memArray = np.frombuffer(bytes, dtype='<f4').copy()
                trace = memArray[0::2] + 1j * memArray[1::2]
                fft_energy = fft(trace)
                energy = np.square(np.abs(fft_energy).astype('f4'))
                energy_sum = np.sum(energy)
                validation_traces[(i - int(key_num * (1.0 - validaion_rate)))][n] = energy_sum
            # validation_traces[(i - int(key_num * (1.0 - validaion_rate)))] /= np.max(validation_traces[(i - int(key_num * (1.0 - validaion_rate)))])
            print('trace :', i)
                #validation_traces[(i-int(key_num*(1.0-validaion_rate)))*16 + n] = energy

    file.close()

    train_group.create_dataset(name = "trace", data = train_traces, dtype = train_traces.dtype)
    valid_group.create_dataset(name = "trace", data = validation_traces, dtype= validation_traces.dtype)
# compute hanmming weight of plaintext and key
def labelize(plaintext, key):
    print(np.float32(AES_Sbox[plaintext[:] ^ key[:]]))
    return np.float32(AES_Sbox[plaintext[:] ^ key[:]])
    """sbox = np.zeros(len(plaintext), dtype='f4')
    for i in range(len(plaintext)):
        sbox[i] = np.float32(AES_Sbox[plaintext[i]^key[i]])
    return sbox"""

if __name__ == "__main__":
    if len(sys.argv)!=2:
        file_path = 'aes_key_100_total_time.raw'
        SAMPLE_RATE = 10000000
        frequency = 80000000
        key_num = 100
        sample_per_key = 100
        record_time = 120897
        key_file_path = "aes_key_100_total_time.txt"
        validation_rate = 0.2
        window_size = 2048
    else:
        file_path = 'aes_1000.raw'
        SAMPLE_RATE = 10000000
        frequency = 80000000
        key_num = 100
        sample_per_key = 100
        record_time = 120897
        validation_rate = 0.2
        #todo : read_parameter

    aes_runtime, exec_runtime, aes_key, plaintext = convert_plaintext_key(key_file_path, key_num)

    saving_path = "AES_key_recover_train.hdf5"

    try:
        output_file = h5py.File(saving_path, 'w')
    except:
        print("Error1:can't create HDF5 file")
        sys.exit(-1)

    train_set_group = output_file.create_group("train_set")
    valid_set_group = output_file.create_group("valid_set")

    convert_raw_to_dataset(file_path, train_set_group, valid_set_group, key_num, sample_per_key, aes_runtime, exec_runtime, window_size,  validation_rate)
    ###
    print("Computing labels")
    ###
    train_index = [n for n in range(0, int(key_num*(1.0 - validation_rate)))]
    valid_index = [n for n in range(int(key_num*(1.0 - validation_rate)), key_num)]
    labels_train = labelize(plaintext[train_index], aes_key[train_index])
    labels_valid = labelize(plaintext[valid_index], aes_key[valid_index])

    ###
    print("Creating output file")
    ###

    train_set_group.create_dataset(name="labels", data=labels_train, dtype=labels_train.dtype)
    valid_set_group.create_dataset(name="labels", data=labels_valid, dtype=labels_valid.dtype)

    metadata_type = np.dtype([("plaintext", plaintext.dtype, (len(plaintext[0]),)),
                              ("key", aes_key.dtype, (len(aes_key[0]),))
                              ])
    train_metadata = np.array([(plaintext[n], aes_key[n]) for n in zip(train_index)], dtype=metadata_type)
    valid_metadata = np.array([(plaintext[n], aes_key[n]) for n in zip(valid_index)], dtype=metadata_type)

    train_set_group.create_dataset(name="metadata", data=train_metadata, dtype=metadata_type)
    valid_set_group.create_dataset(name="metadata", data=valid_metadata, dtype=metadata_type)

    output_file.flush()
    output_file.close()
    try:
        input("Press enter to exit ....")

    except SyntaxError:
        pass