import torch.nn as nn
import torch.nn.functional as F

class LSTM_MODEL(nn.Module):
    def __init__(self,input_size, hidden_dim, target_size = 1, dropout = 0):
        super(LSTM_MODEL, self).__init__()
        self.hidden_dim = hidden_dim
        self.target_size = target_size
        self.dropout = dropout  
        self.input_size = input_size

        ##### Defing Model #####

        self.lstm_1 = nn.LSTM(self.input_size, self.hidden_dim, dropout=self.dropout, bidirectional = True, num_layers=1)

        self.linear_1 = nn.Linear(self.hidden_dim*2, 1)

        
    
    def forward(self, input):
        lstm_out, _ = self.lstm_1(input)
        #print(lstm_out.shape)
        #print(lstm_out.view(2*self.hidden_dim).shape)
        lin_out = self.linear_1(lstm_out.view(len(input), - 1))

        return lin_out

