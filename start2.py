#%%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
WS = 5

# %%
df = pd.read_csv('F:\\ut\\8\\proje\\data\\newbit.csv').dropna()
df = df.iloc[[x for x in range(0,20000,10)]]
# %%
class Bitcoin(nn.Module):
    def __init__(self,inSize, hiddenSize, outSize,ws):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.inSize = inSize
        lstm = []
        for i,item in enumerate(hiddenSize):
            lstm.append(nn.LSTM(inSize,hiddenSize[i],num_layers=2,dropout=.2))
            inSize = hiddenSize[i]
        self.lstm = nn.ModuleList(lstm)
        self.linear = nn.Linear(hiddenSize[-1], outSize)
        self.ws = ws
        self.reset()
    def reset(self):
        self.hiddenMemory =[]
        for item in self.hiddenSize:
            self.hiddenMemory.append((torch.zeros(2,1,item).cuda(),\
                            torch.zeros(2,1,item).cuda()))
    def forward(self,seq):
        seq=seq.view(self.ws,-1,self.inSize)
        for i,item1 in enumerate(self.hiddenMemory):
            seq ,self.hiddenMemory[i]= self.lstm[i](seq, self.hiddenMemory[i])
        x = self.linear(seq)
        return x[-1]
#%%
def make_input(seq,ws):
    output = []
    for item in range(len(seq) - ws):
        output.append( ((seq[item:item+ws].reshape(1,-1)[0].cuda()),\
                        torch.FloatTensor(np.array(seq[item + ws:item + ws+1][-1][-1])).cuda()) )
    return output


# %%
instance = Bitcoin(1,[300,200],1,WS).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(instance.parameters(), lr = .001)
torch.manual_seed(33)
# %%
traincol = ['Open','High','Low', 'Close', 'Volume_(Currency)', 'Weighted_Price']
resultCol = ['Weighted_Price']

#%% 
scaler = MinMaxScaler(feature_range=(-20, 20 ))
train_set = np.array(df[resultCol][-500:-100].values.astype(float)).reshape(-1,1)
train_set = scaler.fit_transform(train_set)
train_set = torch.FloatTensor(train_set)
train_set = make_input(train_set, WS)
test = df[resultCol][-100:].values.astype(float).reshape(-1,1)
test = scaler.fit_transform(test)
test = torch.FloatTensor(test)
test = make_input(test,WS)

# %%
epoch = 30
optimizer = torch.optim.Adam(instance.parameters(), lr=.0001)
for i in range(epoch):
    for seq, y in train_set:
        instance.reset()
        pred = instance.forward(seq)
        loss = criterion(pred,y)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
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
