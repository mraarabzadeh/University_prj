#%%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %%
df = pd.read_csv('F:\\ut\\8\\proje\\data\\bitstampUSD_1-min_data_2012-01-01_to_2019-08-12.csv').dropna()
len(df)
# %%
class Bitcoin(nn.Module):
    def __init__(self,inSize, hiddenSize, outSize,ws):
        super().__init__()
        self.hiddenSize = hiddenSize
        lstm = []
        for i,item in enumerate(hiddenSize):
            lstm.append(nn.LSTM(inSize,hiddenSize[i]))
            inSize = hiddenSize[i]
        self.lstm = nn.ModuleList(lstm)
        self.linear = nn.Linear(hiddenSize[-1], outSize)
        self.ws = ws
        self.reset()
    def reset(self):
        self.hiddenMemory =[]
        for item in self.hiddenSize:
            self.hiddenMemory.append((torch.zeros(1,1,item).cuda(),\
                            torch.zeros(1,1,item).cuda()))
    def forward(self,seq):
        seq=seq.view(self.ws,-1,6)
        for i,item1 in enumerate(self.hiddenMemory):
            seq ,self.hiddenMemory[i]= self.lstm[i](seq, self.hiddenMemory[i])
        x = self.linear(seq)
        return x[-1]
#%%
def make_input(seq,ws,trainCol, resultCol):
    output = []
    for item in range(len(seq) - ws):
        output.append( ((seq[item:item+ws].reshape(1,-1)[0].cuda()),\
                        torch.FloatTensor(np.array(seq[item + ws:item + ws+1][-1][-1])).cuda()) )
    return output


# %%
instance = Bitcoin(6,[100,200],1,20).cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(instance.parameters(), lr = .01)
torch.manual_seed(100)
# %%
traincol = ['Open','High','Low', 'Close', 'Volume_(Currency)', 'Weighted_Price']
resultCol = ['Weighted_Price']

#%% 
train_set = torch.FloatTensor(np.array(df[traincol][-1000:].values.astype(float)).reshape(-1,6))
train_set = make_input(train_set, 20, traincol, resultCol)



# %%
epoch = 30
for i in range(epoch):
    instance.reset()
    for seq, y in train_set:
        instance.reset()
        pred = instance.forward(seq)
        loss = criterion(y, pred)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    print(i, loss)
# %%
torch.save(instance.state_dict(),'start.pn')
# %%
instance = Bitcoin(6,[100,200],1,20)
instance.load_state_dict(torch.load('start.pn'))
instance = instance.cuda()

# %%
instance.eval()
preds = []
real = []
test = torch.FloatTensor(df[traincol][-5000:-20].values.astype(float).reshape(-1,6))
test = make_input(test,20,traincol,resultCol)
with torch.no_grad():
    instance.reset()
    for item, y in test:
        preds.append(instance.forward(item))
        real.append(y)


# %%
# plt.plot(range(len(preds)), preds)
# plt.show()
# plt.plot()
preds
# %%
plt.plot(range(len(real)), real)
plt.show()
# %%
instance.state_dict()

# %%
