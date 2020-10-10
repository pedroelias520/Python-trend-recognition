# -*- coding: utf-8 -*-
"""
Created on Thu Oct  8 22:21:20 2020

@author: Pedro Elias
"""
import numpy as np
import pandas as pd 
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from datetime import datetime
dataset = pd.read_csv('AZUL4.SA.csv')

holidadys_afirm = []
days = [] 
for i in range(len(dataset)):
    data = dataset.iloc[i,0]      
    days.append(data)
    
cal = calendar()
days = np.array(days)
holidays_array = cal.holidays(start='2000-01-03',end='2020-10-08')


for row in range(len(days)):    
    for item in range(len(holidays_array)):        
        if(days[row] == holidays_array[item]):
            holidadys_afirm.append(1)
            print('feriado encontrado!')
            break

print(holidadys_afirm)