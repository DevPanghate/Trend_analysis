from reading_input import createDataFrame, readingInput
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset



def MAKE_DATA(FILENAME, test = False, train =False):
    inputs, channel_length, trade_pairs = readingInput(FILENAME)

    data = createDataFrame(trade_pairs[0])

    data = data.filter(['close'])
    #print(len(data))
    scaler = MinMaxScaler(feature_range= (0, 1))
    scaled_data = scaler.fit_transform(data)
    #print(scaled_data)
    train_data_len = int(len(data)*0.8)
    #print(train_data_len)
    train_data = scaled_data[:train_data_len, :]

    x_train = []
    y_train = []

    for i in range(30, len(train_data)):
        x_train.append(train_data[i - 30:i, 0])
        y_train.append(train_data[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    #print(np.shape(x_train), np.shape(y_train))

    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    #print(np.shape(x_train))

    x_train = torch.Tensor(x_train)

    y_train = torch.Tensor(y_train)

    train = TensorDataset(x_train, y_train)

    train_loader = DataLoader(train, batch_size=1,shuffle = False)
    if test == False:
        return train_loader
    
    test_data = scaled_data[train_data_len - 30:, :]
    #print(len(test_data))
    test_x = []
    test_y = scaled_data[train_data_len:, 0]

    for i in range(30, len(test_data)):
        test_x.append(test_data[i - 30:i, 0])
    
    test_x, test_y = np.array(test_x), np.array(test_y)
    test_x = np.reshape(test_x, (test_x.shape[0], test_x.shape[1], 1))

    test_x = torch.Tensor(test_x)

    test_y = torch.Tensor(test_y)

    test_data = TensorDataset(test_x, test_y)

    test_loader = DataLoader(test_data, batch_size=1,shuffle=False)

    return scaler, train_loader, test_loader
