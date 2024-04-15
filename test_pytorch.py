import random

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

class KeyRecovery(nn.Module):
    def __init__(self):
        super(KeyRecovery, self).__init__()
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=3),
            nn.AvgPool1d(2,stride=1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=3),
            nn.AvgPool1d(2,stride=1),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=3),
            nn.AvgPool1d(2, stride=1),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Linear(1616, 4096)
        self.fc2 = nn.Linear(4096, 256)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.reshape(x, (-1,256))
        x = self.drop(x)
        x = F.softmax(x, dim=0)
        return x


def probability_cal(plaintexts, predictions, number):
    probabilities_array = []

    for i in range(number): # for every traces

        probabilities = np.zeros(256) # one-hot code in 0-255

        for j in range(256):
            hamming_weight = AES_Sbox[plaintexts[i] ^ j]  # hamming weight of every possible keys
            probabilities[j] = predictions[i][hamming_weight]  # probabilities distribution of hamming weight to probabilities distribution of key value

        probabilities_array.append(probabilities) # probabilities of all possible key value (0-255)

    probabilities_array = np.array(probabilities_array)

    return probabilities_array # a nparray of probabilities distribution of key value for all traces


def rank_cal(probabilities, key, number):
    rank = []

    total_probability = np.zeros(256)

    for i in range(number):

        # total probability of every key value
        total_probability = total_probability + np.log(probabilities[i])

        # according to total probability to find the real key value
        sorted_probabilities = np.array(list(map(lambda a: total_probability[a], total_probability.argsort()[::-1])))# the sorted array based on the probabiliy max to min
        real_key_rank = np.where(sorted_probabilities == total_probability[key])[0][0]  # the index of real key value
        rank.append(real_key_rank)

    return np.array(rank)


if __name__ == "__main__":

    test_file_path = "for_testing/3m/10k_d6/1000avg/"

    traces_path = test_file_path + "nor_traces_maxmin.npy"
    key_path = test_file_path + "key.npy"
    plaintext_path = test_file_path + "pt.npy"

    # load file
    traces = np.load(traces_path)
    keys = np.load(key_path)
    plaintexts = np.load(plaintext_path)

    cut = 0
    stop = 10000

    traces = traces[:, [i for i in range(130, 240)]] # AESSBOX(p[0)^k[0])
    traces = traces[cut:stop].astype(np.float32)
    keys = keys[cut:stop]
    plaintexts = plaintexts[cut:stop]

    # choose interest byte of key and plaintext
    interest_byte = 0
    key_interest = keys[interest_byte]
    plaintexts_interest = plaintexts[:, interest_byte]

    # load model parameters

    model_path = "saved_model.pth"
    device = torch.device("cuda")
    net = KeyRecovery()
    net.load_state_dict(torch.load(model_path, map_location="cuda:0"))
    net.to(device)
    net.eval()

    # get prediction of all traces
    traces = torch.from_numpy(traces)
    traces = traces.to(device)
    traces = traces.unsqueeze(1)
    prediction = net(traces).cpu().detach().numpy()
    print(prediction.shape)
    # randomly select trace for testing

    NUMBER = 10000
    average = 100

    rank_array = []

    for i in tqdm(range(100), ncols=60):

        select = random.sample(range(len(traces)), NUMBER)  # [10000,110] -> [10000,1]

        selected_plaintexts_interest = plaintexts_interest[select]
        selected_prediction = prediction[select]

        # calculate subkey probability of selected traces
        probability = probability_cal(selected_plaintexts_interest, selected_prediction, NUMBER)

        # calculate ranks
        ranks = rank_cal(probability, key_interest, NUMBER)

        rank_array.append(ranks)

    # calculate average ranks
    average_ranks = np.sum(np.array(rank_array), axis=0)/average

    # traces number to recover this byte
    for i in range(len(average_ranks)):
        if average_ranks[i] < 0.5:
            print(i)
            break

    np.save('average_ranks.npy', average_ranks)
    plt.plot(average_ranks)
    plt.show()










