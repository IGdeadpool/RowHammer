import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
import os
import sys
import torch.utils.data as Data
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def check_file_exist(file_path):
    if os.path.exists(file_path) == False:
        print("Error: provided file path does not exist")
        sys.exit(-1)
    return


class KeyRecovery(nn.Module):
    def __init__(self):
        super(KeyRecovery, self).__init__()
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(2)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=12),
            nn.AvgPool1d(2, stride=1),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(4, 8, kernel_size=12),
            nn.AvgPool1d(2, stride=1),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(8, 16, kernel_size=12),
            nn.AvgPool1d(2, stride=1),
            nn.LeakyReLU()
        )
        self.fc1 = nn.Linear(1616, 256)
        self.fc2 = nn.Linear(256, 2)
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
        x = torch.reshape(x, (-1, 2))
        x = self.drop(x)
        x = F.softmax(x, dim=0)
        return x


if __name__ == "__main__":

    NUMBER = 100000

    train_file_path = "for_training/cable/100k_d1/100avg/"

    traces_path = train_file_path + "nor_traces_maxmin.npy"
    labels_path = train_file_path + "label_0.npy"

    Traces = np.load(traces_path)

    Traces = Traces[:NUMBER].astype(np.float32)

    Traces = Traces[:, [i for i in range(130, 240)]]
    Traces = torch.from_numpy(Traces)
    labels = np.load(labels_path)
    labels = labels[:NUMBER].astype(np.float32)
    labels = torch.from_numpy(labels)
    print(labels.shape)
    model_file = "saved_model.pth"

    epoches = 20
    batch_size = 128
    validation = 0.1

    train_dataset = Data.TensorDataset(Traces, labels)
    train_data, valid_data = Data.random_split(train_dataset, [round((1 - validation) * Traces.shape[0]),
                                                               round(validation * Traces.shape[0])],
                                               generator=torch.Generator().manual_seed(42))

    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = Data.DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
    print("data finished")

    net = KeyRecovery()
    # use different data to train
    net.load_state_dict(torch.load(model_file, map_location="cuda:0"))
    net.to(device)
    learning_rate = 0.0001
    optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, 10, gamma=1)
    for epoch in range(epoches):
        print("running epoch {}". format(epoch))
        # training
        accuracy_train = 0
        loss_train = 0.0
        net.train()
        for step, (x_train, y_train) in enumerate(train_loader):
            x_train = x_train.to(device)
            x_train = x_train.unsqueeze(1)
            y_train = y_train.to(device)
            output = net(x_train)
            loss = loss_function(output, y_train.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            output_array = output.cpu().detach().numpy()
            accuracy_train += partial_correct_accuracy(y_train.cpu().numpy(), len(y_train), output_array)
            loss_train += loss.data.cpu().numpy()

        print('Epoch: ', epoch+1, '|Batch: ', step,
                '|train loss: %.4f' % (loss_train / (90000//batch_size)),
                '|train accuracy: %.4f' % (accuracy_train / 90000),
                '|learning rate: %.6f' % optimizer.param_groups[0]['lr'])
        accuracy_train = 0
        loss_train = 0.0

        # validation
        net.eval()
        loss_valid = 0.0
        accuracy_valid = 0

        for step, (x_valid, y_valid) in enumerate(valid_loader):
            x_valid = x_valid.to(device)
            x_valid = x_valid.unsqueeze(1)
            y_valid = y_valid.to(device)
            output = net(x_valid)
            loss = loss_function(output, y_valid.long())
            output_array = output.cpu().detach().numpy()

            accuracy_valid += partial_correct_accuracy(y_valid.cpu().numpy(), len(y_valid), output_array)
            loss_valid += loss.data.cpu().numpy()

        print('Epoch: ', epoch + 1,
              '|valid loss: %.4f' % (loss_valid / (10000/batch_size)),
              '|valid accuracy: %.4f' % (accuracy_valid / 10000))

        scheduler.step()
        torch.cuda.empty_cache()

    # save network parameter
    torch.save(net.state_dict(), model_file)
