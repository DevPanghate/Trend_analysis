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


def highs_above(slope, intercept, df, window, start):
    count = 0
    inline = 0
    for i in range(start,start+window):
        if df['high'].iloc[i] > slope*i + intercept:
            count+=1
        elif df['high'].iloc[i] == slope*i + intercept:
            inline+=1
    return count,inline
    


# In[4]:


def lows_below(slope, intercept, df, window, start):
    count = 0
    inline=0
    for i in range(start,start+window):
        if df['low'].iloc[i] < slope*i + intercept:
            count+=1
        elif df['low'].iloc[i] == slope*i + intercept:
            inline+=1
    return count,inline


# In[5]:


def make_channel(df,window,start,step):
    
    #arrays for max and min of wicks
    maxwick = np.array([])
    minwick = np.array([])
    idminwick = np.array([])
    idmaxwick = np.array([])
    
    #arrays for extreme points of window
    maxmin = np.array([])
    ids = np.array([])
    
    #arrays for max and min of candles
    maxcandle=np.array([])
    idmaxcandle=np.array([])
    mincandle=np.array([])
    idmincandle=np.array([])
    
    
    for i in range(start,start+window,step):
        minwick = np.append(minwick, df.low.iloc[i:i+step].min())        
        idminwick = np.append(idminwick, df.low.iloc[i:i+step].idxmin())
        maxwick = np.append(maxwick, df.high.iloc[i:i+step].max())    
        idmaxwick = np.append(idmaxwick, df.high.iloc[i:i+step].idxmax())
        
        maxcandle = np.append(maxcandle, max(df.open.iloc[i:i+step].max(),df.close.iloc[i:i+step].max()))
        if df.open.iloc[i:i+step].max()>df.close.iloc[i:i+step].max():
            idmaxcandle = np.append(idmaxcandle, df.open.iloc[i:i+step].idxmax())
        else:
            idmaxcandle = np.append(idmaxcandle, df.close.iloc[i:i+step].idxmax())
            
         
        mincandle = np.append(mincandle, min(df.open.iloc[i:i+step].min(),df.close.iloc[i:i+step].min()))
        
        if df.open.iloc[i:i+step].min()<df.close.iloc[i:i+step].min():
            idmincandle = np.append(idmincandle, df.open.iloc[i:i+step].idxmin())
        else:
            idmincandle = np.append(idmincandle, df.close.iloc[i:i+step].idxmin())
        
        maxmin = np.append(maxmin, df.low.iloc[i:i+step].min())
        ids = np.append(ids, df.low.iloc[i:i+step].idxmin())
        maxmin = np.append(maxmin, df.high.iloc[i:i+step].max())
        ids = np.append(ids, df.high.iloc[i:i+step].idxmax())
        
    slope, intercept = np.polyfit(ids,maxmin,1)
    intercept_plus_top = (df.high.iloc[idmaxwick] - slope*idmaxwick).max()
    intercept_plus_buttom = (df.high.iloc[idmaxcandle] - slope*idmaxcandle).max()
    
    intercept_minus_buttom = (df.low.iloc[idminwick] - slope*idminwick).min()   
    intercept_minus_top = (df.low.iloc[idmincandle] - slope*idmincandle).min()
    
    plus_diff = intercept_plus_top - intercept_plus_buttom
    minus_diff = intercept_minus_top - intercept_minus_buttom
    
    dec = plus_diff/100
    
    inc = minus_diff/100
    
    for i in range(101):
        intercept_plus_opt = intercept_plus_top - i*dec 
        x,y = highs_above(slope,intercept_plus_opt,df,window,start)
        
        if x>3:
            break
            
    for i in range(101):
        intercept_minus_opt = intercept_minus_buttom + i*inc 
        x,y = lows_below(slope,intercept_minus_opt,df,window,start)
        
        if x>3:
            break
    
        
    
    if intercept_plus_opt-intercept< 0.1* intercept and intercept - intercept_minus_opt< 0.1* intercept:
        dfpl = df[start:start+window]
        fig = go.Figure(data=[go.Candlestick(x=dfpl.index,
                open=dfpl['open'],
                high=dfpl['high'],
                low=dfpl['low'],
                close=dfpl['close'])])
        fig.add_trace(go.Scatter(x=idminwick, y=slope*idminwick + intercept_minus_opt , mode='lines', name='min'))
        fig.add_trace(go.Scatter(x=idmaxwick, y=slope*idmaxwick + intercept_plus_opt, mode='lines', name='max'))
        fig.show()
        
        if not os.path.exists("plots"):
            os.mkdir("plots")
        name  = 'plot' + str(start)    
        fig.write_image("plots/" + name + ".png")
        return True
    return False


# In[ ]:





# In[ ]:




