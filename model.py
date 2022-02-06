from __future__ import division
import pandas as pd
import numpy as np
import datetime
import time
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM

import yfinance as yf
import plotly.graph_objs as go
import pandas_datareader as pdr







def My_model(company_name):
    
    end_date=datetime.date.today()
    year=end_date.year
    month=end_date.month
    day = end_date.day
    if month <= 6 :
        year = year-1
        month = 12-6+month
    else :
        month = month-6
    start_date=datetime.date(year,month,day)
    
    df = pdr.get_data_yahoo(company_name, start=start_date, end=end_date)
    df.drop("Adj Close",axis=1,inplace=True)
    df.to_csv("data.csv")
    df1=df.reset_index()['Close']
    scaler=MinMaxScaler(feature_range=(0,1))
    df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
    
    training_size=int(len(df1)*0.9)
    test_size=len(df1)-training_size
    train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
    
    def create_dataset(dataset, time_step=1):
        dataX, dataY = [], []
        for i in range(len(dataset)-time_step-1):
            a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
            dataX.append(a)
            dataY.append(dataset[i + time_step, 0])
        return np.array(dataX), np.array(dataY)
    time_step =10
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)
    
    X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
    
    model=Sequential()
    model.add(LSTM(50,return_sequences=True,input_shape=(10,1)))
    model.add(LSTM(50,return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error',optimizer='adam')
    model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=60,verbose=0)
    
    train_predict=model.predict(X_train)
    test_predict=model.predict(X_test)
    
    train_predict=scaler.inverse_transform(train_predict)
    test_predict=scaler.inverse_transform(test_predict)
    
    import math
    from sklearn.metrics import mean_squared_error
    MSE_train = math.sqrt(mean_squared_error(y_train,train_predict))
    MSE_test= math.sqrt(mean_squared_error(ytest,test_predict))
    
    x_input=test_data[len(test_data)-10:].reshape(1,-1)
    
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    
    from numpy import array

    lst_output=[]
    n_steps=10
    i=0
    while(i<30):
    
        if(len(temp_input)>10):
            #print(temp_input)
            x_input=np.array(temp_input[1:])
            #print("{} day input {}".format(i,x_input))
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            #print(x_input)
            yhat = model.predict(x_input, verbose=0)
            #print("{} day output {}".format(i,yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            #print(temp_input)
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            #print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            #print(len(temp_input))
            lst_output.extend(yhat.tolist())
            i=i+1
    price_at_30th_day = scaler.inverse_transform([lst_output[-1]]).item(0)
    day_new=np.arange(1,101)
    day_pred=np.arange(101,101+30)
    df3=df1.tolist()
    df3.extend(lst_output)
    df3=scaler.inverse_transform(df3).tolist()
    
    def next_30day_plot():
        plt.plot(df3[len(df3)-30:])
        plt.xlabel("Days")
        plt.ylabel("Price")
    return price_at_30th_day


My_model(company_name)