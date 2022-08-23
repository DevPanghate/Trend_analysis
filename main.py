import torch.nn as nn
import torch
import torch.optim as optim
import pickle
import numpy as np
from model import LSTM_MODEL
from torch.utils.data import DataLoader
from Data import MAKE_DATA
from tqdm import tqdm
from reading_input import readingInput,createDataFrame
from datetime import date
from moving_avergaes import Plot
import plotly
import plotly.graph_objs as go

INPUT_SIZE = 30
HIDDEN_DIM = 80
DROPOUT = 0
LR_RATE = 0.1
MODEL = LSTM_MODEL(input_size=INPUT_SIZE,
                    hidden_dim=HIDDEN_DIM,
                    dropout=DROPOUT)
DEVICE = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu")
OPTIM = optim.SGD(MODEL.parameters(), LR_RATE)
LOSS = nn.MSELoss(reduction="mean")
FILENAME = "inputfile.txt"
Scaler, Train_Data, Test_Data = MAKE_DATA(FILENAME, test = True, train = False)
NUM_TRAINING_STEPS = len(Train_Data)*15
progress_bar = tqdm(range(NUM_TRAINING_STEPS), position=0, leave = True)


def Training(MODEL, OPTIM, LOSS, DATA, EPOCHS, DEVICE, GPU = False, SAVE = False):
    if GPU:
        MODEL = MODEL.to(DEVICE)
        LOSS = LOSS.to(DEVICE)

    MODEL.train()

    print(f"The Size of the data is - {len(DATA)}")
    print(EPOCHS, len(DATA))
    for epoch in range(EPOCHS):
        batch_loss = []
        for train, pred in DATA:
            train = train.to(DEVICE).view(1, 1, 30)
            #print(train.shape)
            #print(torch.Size(train.))
            pred = pred.to(DEVICE)

            yhat = MODEL(train)[0]
            loss = LOSS(pred, yhat)
            #print(pred, yhat)
            batch_loss.append(loss.item())
            loss.backward()
            
            OPTIM.step()

            OPTIM.zero_grad()
            progress_bar.update()
            #break
        print(np.mean(batch_loss))
    if SAVE:
        torch.save(MODEL.state_dict(), "Model_State_Dict_15.pt")
def Testing(MODEL, DATA,DEVICE, SCALER, LOSS):
    with torch.no_grad():
        predictions = []
        values = []
        MODEL.to(DEVICE)
        MODEL.eval()
        for x_test, y_test in DATA:
            x_test = x_test.to(DEVICE).view(1, 1, 30)
            y_test = y_test.to(DEVICE)

            yhat = MODEL(x_test)[0]

            predictions.append(yhat.cpu().detach().numpy())
            values.append(y_test.cpu().detach().numpy())
        
        #predictions = SCALER.inverse_transform(predictions)
        #loss = LOSS(values, predictions)
        rmse = np.sqrt(np.mean(((np.array(predictions) - np.array(values)) ** 2)))
        #print(f"Test-Loss - {loss}")
        print(f"RMSE - Loss - {rmse}")
        return predictions, values

def Plot_Graph(yhat, ytest):
    inputs, channel_length, trade_pair = readingInput(FILENAME)
    df = createDataFrame(trade_pair[0])
    df['Date'] = df['unix_timestamp'].map(lambda x: date.fromtimestamp(x))
    print(len(yhat))
    yhat = np.append(df.close[:-74], yhat)
    ytest = np.append(df.close[:-74], ytest)
    print(len(yhat), len(ytest))
    fig = Plot(df, yhat, ytest)
    plotly.offline.plot(
    { "data": fig,"layout": go.Layout(title = "hello world")}, auto_open = True)
if __name__ == "__main__":
    try:
        MODEL.load_state_dict(torch.load("Model_State_Dict_15.pt"))
        print('Saved Model Found!')
    except:
        Training(MODEL, OPTIM, LOSS, Train_Data, 15,
            DEVICE, GPU = True, SAVE=True)
    predictions, values = Testing(MODEL,Test_Data, 
                     DEVICE, Scaler, LOSS)

    predictions, values = Scaler.inverse_transform(predictions), Scaler.inverse_transform(values)
    predictions, values = np.reshape(predictions, predictions.shape[0]), np.reshape(values, values.shape[0])
    #print(len(predictions), len(values))
    #print(predictions)
    Plot_Graph(predictions,values)