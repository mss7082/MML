import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Data Import
from pandas_datareader import data as pdr
import yfinance as yf


#Modelling
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


#Modelling Metrics
from sklearn import metrics



yf.pdr_override()


df_full = pdr.get_data_yahoo("AMZN", start="2017-01-01").reset_index()
df_full.to_csv("amzn.csv", index=False)

print(df_full.shape)
print(df_full.head(10))

df_full.set_index("Date", inplace=True)
print(df_full.head())

window_size=32
num_samples = len(df_full)-window_size


#Get indices of access for the data
indices=np.arange(num_samples).astype(np.int)[:,None]+np.arange(window_size+1).astype(np.int)


data = df_full["Adj Close"].values[indices] #Create the 2D matric of training
x = data[:,:-1] # Each row represents 32 days in the past
y = data[:,-1] #Each output value represents the 33rd day

split_fraction=0.8
ind_split=int(split_fraction*num_samples)

x_train = x[:ind_split]
y_train = y[:ind_split]
x_test = x[ind_split:]
y_test = y[ind_split:]


print(x_train.shape)


#Help Functions
def get_performance (model_pred):
  #Function returns standard performance metrics
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, model_pred).round(4))  
  print('Mean Squared Error:', metrics.mean_squared_error(y_test, model_pred).round(4))  
  print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, model_pred)).round(4))
  
  
  
def get_plot (model_pred):
  plt.scatter(model_pred, y_test, color="gray")
  plt.plot(y_test, y_test, color='red', linewidth=2)


y_pred_lag = np.roll(y_test, 1)
get_performance(y_pred_lag)

get_plot(y_pred_lag)

model_lr = LinearRegression()
model_lr.fit(x_train, y_train)

y_pred_lr = model_lr.predict(x_test)
get_performance(y_pred_lr)


get_performance(y_pred_lr)


model_ridge = Ridge()
model_ridge.fit(x_train, y_train)

#prediction
y_pred_ridge=model_ridge.predict(x_test)

get_performance(y_pred_ridge)


get_plot(y_pred_ridge)


model_gb = GradientBoostingRegressor()
model_gb.fit(x_train, y_train)

y_pred_gb = model_gb.predict(x_test)

get_performance(y_pred_gb)
get_plot(y_pred_gb)


df_comp=pd.DataFrame({"lag":np.absolute(y_test-y_pred_lag), 
              "lr":np.absolute(y_test-y_pred_lr), 
              "ridge":np.absolute(y_test-y_pred_ridge),
              "gb":np.absolute(y_test-y_pred_gb)})


df_comp.plot.bar(figsize=(16, 6))
plt.ylim(0,20)
plt.xlim(11,20)