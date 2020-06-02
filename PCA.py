#%%
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#%%
df = pd.read_csv('F:\\ut\\8\\proje\\data\\newbit.csv').dropna()
df = df.iloc[[x for x in range(0,20000,10)]]
df

#%%
X_train, X_test, y_train, y_test = train_test_split(df, df['Close'], test_size=0.2, random_state=0)


#%%
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# %%
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# %%
if __name__ == "__main__":
    