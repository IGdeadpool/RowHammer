import sys
from tqdm import tqdm

import h5py
import numpy as np

if __name__ == "__main__":
    saving_path = "AES_key_recover_train.hdf5"
    original_file_path = "rawhammer_db/True.csv"

    try:
        output_file = h5py.File(saving_path, 'w')
    except:
        print("Error1:can't create HDF5 file")
        sys.exit(-1)

    train_set_group = output_file.create_group("train_set")
    valid_set_group = output_file.create_group("valid_set")

    train_index = [n for n in range(0,800)]
    valid_index = [n for n in range(800, 200)]

    ####
    print("extract traces")
    ###
    data = np.loadtxt(open(original_file_path))



    raw_traces = input_file['traces']
    raw_data = input_file['metadata']

    raw_plaintext = raw_data['plaintext']
    raw_keys = raw_data['key']

    min_target_point = min(target_points)
    max_target_point = max(target_points)

    print(min_target_point,max_target_point)
    target_points = np.array(target_points)

    ###
    print("Processing train traces...")
    ###
    raw_traces_train = np.zeros([len(train_index), len(target_points)], raw_traces.dtype)
    current_trace = 0
    for trace in tqdm(train_index):
        raw_traces_train[current_trace] = raw_traces[trace, min_target_point:max_target_point+1]
        current_trace += 1

    ###
    print("Processing validation traces...")
    ###
    raw_traces_valid = np.zeros([len(valid_index), len(target_points)], raw_traces.dtype)
    current_trace = 0
    for trace in tqdm(valid_index):
        raw_traces_valid[current_trace] = raw_traces[trace, min_target_point:max_target_point+1]
        current_trace += 1

    ###
    print("Computing labels")
    ###
    labels_train = labelize(raw_plaintext[train_index], raw_keys[train_index])
    labels_valid = labelize(raw_plaintext[valid_index], raw_keys[valid_index])


    ###
    print("Creating output_file...")
    ###

    train_set_group.create_dataset(name="traces", data=raw_traces_train, dtype=raw_traces_train.dtype)
    valid_set_group.create_dataset(name="traces", data=raw_traces_valid, dtype=raw_traces_valid.dtype)

    train_set_group.create_dataset(name="labels", data= labels_train, dtype=labels_train.dtype)
    valid_set_group.create_dataset(name="labels", data= labels_valid, dtype=labels_valid.dtype)

    output_file.flush()
    output_file.close()
    input_file.close()