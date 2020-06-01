# %%
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf #ar --> pcf, ma-->acf
from statsmodels.tsa.stattools import adfuller
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#%% 
df = pd.read_csv('data/bitcoin_csv.csv').dropna()
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')
df

# %% 
def check_adfuler(dataframe):
  result = adfuller(dataframe)
  label = ['adf', 'p_val', 'lags', 'num_of_observ']
  for i , item in zip(label,result):
    print(i, ':', item)
  if result[1] <= .05:
    print('stationary')
  else:
    print('none stationary')

#%%
new_df = df['price(USD)'].fillna(0)
new_df = pd.to_numeric(new_df,errors='coerce')
check_adfuler(new_df)

#%%
fig = plt.figure(figsize=(8,12))
ax1 = fig.add_subplot(211)
fig = plot_acf(new_df, lags=40, ax = ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(new_df, lags=40,ax = ax2)


#%%
new_df = new_df.diff(1).dropna()
train = new_df[-200:-50]
test = new_df[-50:]
#%%
best_ord = (0,0,0)
best_aic = 999999999
for i in range(15):
  for j in range(15):
    for k in range(15):
      try:
        model = ARIMA(train,order=(i,j,k)).fit()
        print(model.aic)
        if model.aic < best_aic:
          best_ord = (i,j,k)
          best_aic = model.aic
      except:
        continue
print(best_ord)
print('\n\n\n\n\n\n')
# %%
model = ARIMA(train,order=best_ord).fit()
predicts= model.predict(start=150,end=160)
predicts*=-1
predicts.plot()
test.plot()

# %%
predict = model.forecast(steps=50)
predict[0]
plt.plot(test)
plt.show()
plt.plot(predict[0])
# %%
