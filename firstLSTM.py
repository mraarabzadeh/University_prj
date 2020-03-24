#%%
import torch
import torch.nn as nn
import numpy as np
import pandas as pd

#%%
df = pd.read_csv('Alcohol_Sales.csv')
df

# %%
class Model(nn.Module):
    def __init__(self,inp_size,hiddenSize,output):
        super().__init__()
        self.hiddenSize = hiddenSize
        self.lstm = nn.LSTM(inp_size,hiddenSize)
        self.linear = nn.Linear(hiddenSize,output)
        self.reset()
    def reset(self):
        self.hiddenArr = (torch.zeros(1,1,self.hiddenSize) , torch.zeros(1,1,self.hiddenSize))
    def forward(self, inputArr ):
        lstm , self.hiddenArr = self.lstm(inputArr.view(len(inputArr),-1,1),self.hiddenArr)
        lstm = self.linear(lstm.view(len(lstm),-1))
        return lstm[-1]
model = Model(1,50,1)

# %%
def makeSequence(seq,windowSize):
    output = []
    for i in range(len(seq) - windowSize):
        output.append((seq[i:i + windowSize].reshape(1,-1)[0], seq[i+windowSize:i+windowSize+1]))
    return output

# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1,2))
train_data = df['S4248SM144NCEN'].values.astype(float)[:-12]
test_data = torch.FloatTensor(df['S4248SM144NCEN'].values.astype(float)[-12:]).view(-1)

train_data = scaler.fit_transform(train_data.reshape(-1,1))
train_data = torch.FloatTensor(train_data)
train_data = makeSequence(train_data,12)
train_data[0]
# %%
torch.manual_seed(33)

criteriion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)
epoch = 30



# %%
losses = []
for i in range(epoch):
    for seq, y_train in train_data:
        optimizer.zero_grad()
        model.reset()

        pred = model.forward(seq)
        loss = criteriion(pred, y_train)
        loss.backward()
        optimizer.step()
    losses.append(loss)
    
    if i % 10 == 1:
        print('epochs:{}'.format(i))

# %%
import matplotlib.pyplot as plt
plt.plot(range(len(losses)), losses)
plt.show()
# %%
# torch.save(model.state_dict(),'result.pn')
model = Model(1,50,1)
model.load_state_dict(torch.load('result.pn'))
# %%
pred = df['S4248SM144NCEN'].values.astype(float)
pred = scaler.fit_transform(pred.reshape(-1,1)).tolist()
model.eval()
for i in range(12):
    seq = torch.FloatTensor(pred[-12:])
    with torch.no_grad():
        model.reset()
        pred.append(model.forward(seq))
# %%
pred[-12:]
result = scaler.inverse_transform(np.array(pred[-12:]).reshape(-1,1))
# %%
plt.plot(range(12), result)
plt.show()
plt.plot(range(12),df['S4248SM144NCEN'][-12:])
plt.show()

# %%
