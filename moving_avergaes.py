#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gate_api
from gate_api.exceptions import ApiException, GateApiException


# In[2]:


import pandas as pd
import numpy as np
import datetime
from datetime import datetime
from datetime import date
import time
import plotly
import plotly.graph_objects as go
from matplotlib import pyplot
import os


# In[3]:


from channel_formation import make_channel, highs_above, lows_below


# In[4]:


configuration = gate_api.Configuration(
    host = "https://api.gateio.ws/api/v4"
)


api_client = gate_api.ApiClient(configuration)
# Create an instance of the API class
api_instance = gate_api.SpotApi(api_client)


# In[5]:


def readingInput(filename):    
    input_file = open(filename,'r')
    inputs = input_file.readline()
    inputs = int(inputs)
    if inputs == 0:
        print('number of inputs is 0')
        return
    
    channel_length = input_file.readline()
    input_file.seek(0)
    trade_pairs=[]
    x = input_file.readlines()
    #print(trades)
    #for i in range(len(x)-1):
        
        #x[i]=x[i][:-2]
    x = x[2:]
    for i in x:
        trade_pairs.append(i.split(sep=', '))
    return inputs,channel_length,trade_pairs


# In[6]:


inputs,channel_length,trade_pairs = readingInput('inputfile.txt')
#print(trade_pairs)


# In[7]:


def createDataFrame(trade_info):
    
    currency_pair = trade_info[0]+'_USDT' 
    
    time1 = trade_info[2].split(sep='/')
    if trade_info[3][-1]=='\n':
        time2 = trade_info[3][:-1]
    else:
        time2 = trade_info[3]
    #print(time2)
    time2= time2.split(sep='/')
    epoch = date(1970, 1, 1)
    date1 = date(int(time1[-1]),int(time1[-3]),int(time1[-2]))
    date2 = date(int(time2[-1]),int(time2[-3]),int(time2[-2]))
    diff1 = date1-epoch
    diff2 = date2-epoch
    
    ts1 =  int(diff1.total_seconds())
    ts2 = int(diff2.total_seconds())
    #print(date1,date2)
    #print(ts1,ts2)
    _from = ts1 
    to = ts2
    interval = trade_info[1]
    interval_map={'10s':10,'5m':300,'10m':600,'15m':900,'30m':1800,'1h':3600,'4h':14400,'8h':28800,'1d':86400,'7d':604800}
    
    if interval_map.get(interval)==None:
        print('Invalid interval')
        return
    if (ts2-ts1)/interval_map[interval] > 1000 :
        print(ts2-ts1/interval_map[interval])
        print('number of points exceeded')
        return
    if (ts2-ts1)/interval_map[interval] < 1 :
        print('number of points less than 1')
        return
    
    
    try:
        # Market candlesticks
        api_response = api_instance.list_candlesticks(currency_pair, _from=_from , to = to, interval=interval)

        #print(api_response)
    except GateApiException as ex:
        print("Gate api exception, label: %s, message: %s\n" % (ex.label, ex.message))
    except ApiException as e:
        print("Exception when calling SpotApi->list_candlesticks: %s\n" % e)
    for i in api_response:
        
        i = i.remove(i[6])
    #print(api_response)
    df = pd.DataFrame(api_response, columns=['unix timestamp','trading volume','close','high','low','open'])
    
    df['unix_timestamp'] = df['unix timestamp'].astype('int')
    df['trading_volume'] = df['trading volume'].astype('float64')
    df['close'] = df['close'].astype('float64')
    df['open'] = df['open'].astype('float64')
    df['low'] = df['low'].astype('float64')
    df['high'] = df['high'].astype('float64')
    return df


# In[8]:


df = createDataFrame(trade_pairs[0])
df['Date'] = df['unix_timestamp'].map(lambda x: date.fromtimestamp(x))
df


# In[9]:


def SMA(days,df):
    avgs = df.close.rolling(days).mean()
    try:
        fig.add_traces([go.Scatter(x=df.Date , y = avgs)])
        
    except:
        fig = go.Figure(data=[go.Candlestick(x=df.Date,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'])])
        fig.add_traces(go.Scatter(x=df.Date , y = avgs))
      
        
    


# In[15]:


def EMA(days,df):
    k = 2/(days+1)
    avgs = np.array([np.nan]*(days-1))
    avgs = np.append(avgs,df.close[:days].mean())
    for i in range(days,len(df)):
        avgs = np.append(avgs,k*(df.close.iloc[i]-avgs[i-1]) + avgs[i-1])
        
    return avgs


# In[16]:





# In[ ]:




