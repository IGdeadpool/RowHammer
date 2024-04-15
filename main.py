import sys
from tqdm import tqdm
import torch
import torch.utils.data as Data

from torch.utils.data import DataLoader, Dataset, TensorDataset
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RawhammerDetection(nn.Module):
    def __init__(self):
        super(RawhammerDetection, self).__init__()
        self.bn1 = nn.BatchNorm1d(2048)
        self.bn2 = nn.BatchNorm1d(1)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=12, stride=1, padding=4),
            nn.LeakyReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=12, stride=1, padding=4),
            nn.LeakyReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=12, stride=1, padding=4),
            nn.AvgPool2d(2, 1),
            nn.LeakyReLU()
        )
        self.fc1 = nn.LazyLinear(2048)
        self.fc2 = nn.LazyLinear(2)
        self.drop = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))
        #x = self.bn1(x)
        x = self.fc2(x)
        #x = self.bn2(x)
        x = torch.reshape(x, (-1,2))
        x = self.drop(x)
        x = F.softmax(x, dim=1)
        #x = F.softmax(x, dim=0)
        return x

class dataset_prediction(Dataset):

    def __init__(self, data, label):
        self.len = len(data)
        self.x_data = torch.from_numpy(data)
        self.y_data = torch.from_numpy(label)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

if __name__ == "__main__":
    saving_path = "AES_key_recover_train.hdf5"
    original_file_path = "rawhammer_db/True.csv"
    original_label_path = "rawhammer_db/labels.csv"

    data = np.loadtxt(original_file_path, delimiter=',', dtype=np.float32)
    labels = np.loadtxt(original_label_path, delimiter=',', dtype=np.float32)  # 使用numpy读取数据

    data = data.reshape(4000, 12, 8192)

    train_set = dataset_prediction(data=data, label=labels)
    #print(train_set.shape)

    epoches = 20
    batch_size = 20
    validation = 0.1

    train_data, valid_data = Data.random_split(train_set, [round((1 - validation) * data.shape[0]),
                                                               round(validation * data.shape[0])],
                                               generator=torch.Generator().manual_seed(42))

    train_loader = Data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    valid_loader = Data.DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True)
    print("data finished")

    net = RawhammerDetection()
    # use different data to train
    #net.load_state_dict(torch.load(model_file, map_location="cuda:0"))
    net.to(device)
    learning_rate = 0.0001
    optimizer = torch.optim.RMSprop(net.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, 10, gamma=1)
    for epoch in range(epoches):
        print("running epoch {}".format(epoch))
        # training
        accuracy_train = 0.0
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

            accuracy_train += np.sum(y_train.data.cpu().numpy()==np.argmax(output.data.cpu().numpy(), axis =1))
            loss_train += loss.data.cpu().numpy()

        print('Epoch: ', epoch + 1, '|Batch: ', step,
              '|train loss: %.4f' % (loss_train / (2700 // batch_size)),
              '|train accuracy: %.4f' % (accuracy_train / 2700),
              '|learning rate: %.6f' % optimizer.param_groups[0]['lr'])
        accuracy_train = 0.0
        loss_train = 0.0

        # validation
        net.eval()
        loss_valid = 0.0
        accuracy_valid = 0.0

        for step, (x_valid, y_valid) in enumerate(valid_loader):
            x_valid = x_valid.to(device)
            x_valid = x_valid.unsqueeze(1)
            y_valid = y_valid.to(device)
            output_valid = net(x_valid)
            loss = loss_function(output_valid, y_valid.long())
            accuracy_valid += np.sum(y_valid.data.cpu().numpy()==np.argmax(output_valid.data.cpu().numpy(), axis=1))
            loss_valid += loss.data.cpu().numpy()

        print('Epoch: ', epoch + 1,
              '|valid loss: %.4f' % (loss_valid / (300 / batch_size)),
              '|valid accuracy: %.4f' % (accuracy_valid / 300))

        scheduler.step()
        torch.cuda.empty_cache()






