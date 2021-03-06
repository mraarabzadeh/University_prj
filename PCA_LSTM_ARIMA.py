#%%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
# from skcuda.linalg import PCA as cuPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import EMAIndicator
from ta.momentum import UltimateOscillator, StochasticOscillator, ROCIndicator, RSIIndicator, WilliamsRIndicator
from statsmodels.tsa.arima_model import ARIMA

WS = 5

# %%
df = pd.read_csv('F:\\ut\\8\\proje\\data\\newbit.csv').dropna()
df = df.iloc[[x for x in range(0,20000,10)]]
# %%
class Bitcoin(nn.Module):
    def __init__(self,inSize, hiddenSize, outSize, ws):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.inSize = inSize
        self.ws = ws
        lstm = []
        for i,item in enumerate(hiddenSize):
            lstm.append(nn.LSTM(inSize,hiddenSize[i],num_layers=2,dropout=.2))
            inSize = hiddenSize[i]
        self.lstm = nn.ModuleList(lstm)
        self.linear = nn.Linear(hiddenSize[-1], outSize)
        self.reset()
    def reset(self):
        self.hiddenMemory =[]
        for item in self.hiddenSize:
            self.hiddenMemory.append((torch.zeros(2,1,item).cuda(),\
                            torch.zeros(2,1,item).cuda()))
    def forward(self,seq):
        seq=seq.view(self.ws, -1, self.inSize)
        for i,item1 in enumerate(self.hiddenMemory):
            seq ,self.hiddenMemory[i]= self.lstm[i](seq, self.hiddenMemory[i])
        x = self.linear(seq)
        return x[-1]
#%%
def make_input(seq,ws, st_col=0, end_col=0):
    output = []
    for item in range(len(seq) - ws):
        if st_col == end_col == 0:
            output.append( ((seq[item:item+ws].reshape(1,-1)[0].cuda()),\
                            torch.FloatTensor(np.array(seq[item + ws:item + ws+1][-1][-1])).cuda()) )
        else:
            output.append(((torch.FloatTensor(pd.DataFrame([x[st_col:end_col] for x in seq[item:item+ws]]).values.astype(float).reshape(1,-1)[0]).cuda()),\
                            torch.FloatTensor(np.array(seq[item + ws:item + ws+1][-1][-1])).cuda()))
    return output


#%%
def PCA_function(data_frame, n_components):
    sc = StandardScaler()
    X_train = sc.fit_transform(data_frame)
    pca = PCA(n_components=n_components)
    X_train = pca.fit_transform(X_train)
    return X_train


#%%
def arima_model(data):
    return ARIMA(data, order=(1,1,1)).fit().forecast(steps=1)[0]


#%% 
def add_indicators():
    added_columns = ['MA5','MA10','MA20', 'DIFF', 'BU', 'BL', 'Stochastic', \
                 'ROC', 'RSI6', 'RSI12', 'ATR', 'WR5', 'WR10', 'UOS'\
                     'I30', 'I31', 'I32', 'I33', 'I34', 'I35', 'I36', \
                      'I28', 'I29'   ]
    close = df['Close']
    df['MA5'] = close.rolling(window=5).mean()
    df['MA10'] = close.rolling(window=10).mean()
    df['MA20'] = close.rolling(window=20).mean()
    df['DIFF'] = EMAIndicator(close, n=12).ema_indicator() - EMAIndicator(close, n=26).ema_indicator()
    df['BL']=BollingerBands(close, n=20, ndev=2).bollinger_lband()
    df['BU']=BollingerBands(close, n=20, ndev=2).bollinger_hband()
    df['Stochastic']=StochasticOscillator(df['High'], df['Low'], close).stoch()
    df['ROC'] = ROCIndicator(close).roc()
    df['RSI6'] = RSIIndicator(close, 6).rsi()
    df['RSI12'] = RSIIndicator(close, 12).rsi()
    df['ATR'] = AverageTrueRange(df['High'],df['Low'], close).average_true_range() 
    df['WR10'] = WilliamsRIndicator(df['High'],df['Low'], close, lbp=10).wr()
    df['WR5'] = WilliamsRIndicator(df['High'],df['Low'], close, lbp=5).wr()
    df['UOS'] = UltimateOscillator(df['High'],df['Low'], close,).uo()
    df['I28'] = close.diff(1) / close.shift(1)
    df['I29'] = (close - df['Open']) / df['Open']
    df['I30'] = (close - df['High']) / (df['High'] - df['Low'])
    df['I31'] = df['MA5'].diff(1) / df['MA5'].shift(1)
    df['I32'] = df['MA10'].diff(1) / df['MA10'].shift(1)
    df['I33'] = df['MA20'].diff(1) / df['MA20'].shift(1)
    df['I34'] = df['MA5'].diff(1) / df['MA20'].shift(1)
    df['I35'] = (close - np.array([np.amin(close[:x]) for x in range(len(close))]))/np.array([np.amin(close[:x]) for x in range(len(close))])
    df['I36'] = (close - np.array([np.amax(close[:x]) for x in range(len(close))]))/np.array([np.amax(close[:x]) for x in range(len(close))])
    df.dropna(inplace= True)
    return added_columns


# %%
instance = Bitcoin(10,[300,200],1,WS).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(instance.parameters(), lr = .001)
torch.manual_seed(33)
# %%
traincol = ['Open','High','Low', 'Close', 'Volume_(Currency)', 'Weighted_Price']
resultCol = ['Weighted_Price']


#%% 
scaler = MinMaxScaler(feature_range=(-1, 1 ))
add_indicators()
new_df = np.array(PCA_function(df, 10))
new_df=np.append(new_df, df['Close'].values.astype(float).reshape(-1,1), axis=1)
new_df = pd.DataFrame(data=new_df)
train_set = np.array(new_df.values.astype(float))#.reshape(-1,1)
train_set = scaler.fit_transform(train_set)
# train_set = torch.FloatTensor(train_set)
train_set = make_input(train_set, WS, st_col=0, end_col=10)
# test = df[resultCol][-100:].values.astype(float)#.reshape(-1,1)
# test = scaler.fit_transform(test)
# test = torch.FloatTensor(test)
# test = make_input(test,WS)
test = train_set[-200:]
# %%
epoch = 200
preds_in_learn = []
reals_in_learn = []
optimizer = torch.optim.Adam(instance.parameters(), lr=.000001)
for i in range(epoch):
    for seq, y in train_set:
        instance.reset()
        pred = instance.forward(seq)
        loss = criterion(pred,y)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    with torch.no_grad():
        for item , y in test:
            preds_in_learn.append(instance.forward(item))
            reals_in_learn.append(y)
    plt.plot(range(50), preds_in_learn[-50:],)
    # plt.show()
    plt.grid(color='r', linestyle='-', linewidth=.1, zorder=.1)
    # plt.show()
    plt.plot(range(50), reals_in_learn[-50:])
    plt.show()
    preds_in_learn.clear()
    reals_in_learn.clear()
    print(i, loss)
# %%
torch.save(instance.state_dict(),'start400_5.pn')
# %%
instance = Bitcoin(1,[100,200],1,WS)
instance.load_state_dict(torch.load('start1000_360.pn'))
instance = instance.cuda()

# %%
# instance.eval()
preds = []
real = []
with torch.no_grad():
    for item, y in test:
        instance.reset()

        preds.append(instance.forward(item))
        real.append(y)


# %%
# torch.tensor().item()
# print(preds)
# preds = [torch.tensor(x).item() for x in preds]
# plt.scatter(np.linspace(10000,17000,len(preds)),preds)
plt.plot(range(94), preds[-94:],)
# plt.show()
plt.plot(range(94), real[-94:])
plt.show()
# plt.plot()
# preds = set(preds)
# preds = set(preds)
# %%
plt.plot(range(len(real)), real)
plt.show()
# %%
instance

# %%


# %%
