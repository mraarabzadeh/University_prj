import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator
from ta.momentum import UltimateOscillator, StochasticOscillator, ROCIndicator, RSIIndicator, WilliamsRIndicator
WS = 20
class Bitcoin(nn.Module):
    def __init__(self,inSize, touples_index, variant_size, hiddenSize, outSize, ws):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.inSize = inSize
        self.ws = ws
        self.touples_index = touples_index
        varient_lstm = []
        lstm = []
        for i, item in enumerate(variant_size):
            varient_lstm.append(nn.LSTM(2,item, num_layers=2, dropout=.3))
            varient_lstm.append(nn.LSTM(item, 1, num_layers=1, dropout=.2))
        self.varient_lstm = nn.ModuleList(varient_lstm)
        self.variant_size = variant_size
        for i,item in enumerate(hiddenSize):
            lstm.append(nn.LSTM(inSize,hiddenSize[i],num_layers=1,dropout=.2))
            inSize = hiddenSize[i]
        self.lstm = nn.ModuleList(lstm)
        self.linear = nn.Linear(hiddenSize[-1], outSize)
        self.reset()
        
    def reset(self):
        self.hiddenMemory =[]
        self.varient_mem = []
        for item in self.variant_size:
            self.varient_mem.append((torch.zeros(2,1,item).cuda(), torch.zeros(2,1,item).cuda()))
            self.varient_mem.append((torch.zeros(1,1,1).cuda(), torch.zeros(1,1,1).cuda()))
            
        for item in self.hiddenSize:
            self.hiddenMemory.append((torch.zeros(1,1,item).cuda(),\
                            torch.zeros(1,1,item).cuda()))
    def forward(self,seq):
        var_x = seq[1]
        var_result = [0 for x in range(len(var_x))]
        for i in range(0, len(self.varient_mem), 2):
            x ,self.varient_mem[i]= self.varient_lstm[i](var_x[int(i/2)], self.varient_mem[i])
            var_result[int(i/2)] ,self.varient_mem[i+1]= self.varient_lstm[i+1](x, self.varient_mem[i+1])
        
        x = torch.cat(var_result, 2)
        seq = torch.cat((seq[0], x), 2)

        for i,item1 in enumerate(self.hiddenMemory):
            seq ,self.hiddenMemory[i]= self.lstm[i](seq, self.hiddenMemory[i])
        x = self.linear(seq)
        return x[-1]

def make_input(seq, ws, touples, st_col=0, end_col=0):
    output = []
    for item in range(len(seq) - ws):
        l = []
        for item2 in touples:
                l.append(torch.FloatTensor(seq[item:item+ws][ [item2[0], item2[1]] ].values.astype(float).reshape(ws, -1, 2)).cuda())
        
        not_used_col =set(range(st_col, end_col))
        for i in touples:
            not_used_col.remove(i[0])
            not_used_col.remove(i[1])

        not_used_col = list(not_used_col)
        output.append( ( (torch.FloatTensor( seq[item:item+ws][not_used_col].values.astype(float).reshape(ws, -1, len(not_used_col)) ).cuda(), l),\
                        torch.FloatTensor(np.array(seq[item + ws:item + ws+1][end_col])).cuda()) )
    return output
def inputs(df):
    scaler = MinMaxScaler(feature_range=(-1, 1 ))
    # add_indicators()
    traincol = ['Open','High','Low', 'Close', 'Volume_(Currency)', ]

    new_df=np.append(df[traincol], df['Close'].values.astype(float).reshape(-1,1), axis=1)
    new_df = pd.DataFrame(data=new_df)

    train_set = np.array(new_df[-500:-100].values.astype(float))
    train_set = scaler.fit_transform(train_set)
    train_set = pd.DataFrame(train_set)
    train_set = make_input(train_set, WS, [(1,2), (3,4)], st_col=0, end_col=5)

    test = new_df[-105:].values.astype(float)
    test = scaler.fit_transform(test)
    test = pd.DataFrame(test)
    test = make_input(test,WS,[(1,2), (3,4)], st_col=0, end_col=5)
    return train_set, test