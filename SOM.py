#!/usr/bin/env python
# coding: utf-8

# ## Time Series Forecasting with Recurrent Neural Network (RNN)

# by Haydar Özler and Tankut Tekeli

# ### Abstract

# * Purpose: to practice solving a timeseries problem by using Recurrent Neural Network
# 
# * Data: Bike Sharing in Washington D.C. Dataset
# 
# * Applied Tools & Methods: TimeSeriesGenerator, SimpleRNN
# 
# * Result: Around %80 correctness (calculated as 1-mae/mean)
# 
# * Further Studies: More advanced sequential methods like LSTM and GRU can be applied.

# ### Explanation of the Study

# We have created a model to predict how many bicycles will be rented in the following days. The features used like weather, temperature, working day are explained in the following sections in detail. 
# 
# We have used SimpleRNN method in Keras library. It is one of the sequential models. The others are LSTM and GRU. 
# 
# Sequential models have 3 dimension (sample size, time steps, features). Preparing 3D input is another challenge. Instead of trying to create a 3D array, we use TimeSeriesGenerator class which brings some other advantages like setting the batch size.
# 
# We skipped feature engineering and visualization parts because main purpose was to practice a sequential neural network. It is possible to have better achivements by applying these methods and then create a predictive model. 
# 
# Data is 2 years daily data. Number of samples is 731. We have splitted it into 631, 50, 50 as train, test and hold-out data respectively.
# 
# We have measured the performance of the model with ( 1 - (mean average error) / (mean) ) and we have reached values around %80.
# 
# There are so many further studies: More feature engineering for better accuracy and trying other sequential models. 

# ### Importing Libraries

# In[1]:


import pandas as pd;
import matplotlib.pyplot as plt
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import RNN, SimpleRNN
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.layers import Dropout
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.callbacks import LambdaCallback
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import seaborn as sns


# ### Data Preprocessing

# #### Reading the dataset

# In[2]:


dataset = pd.read_csv('VALE3.SA.csv')


# In[3]:


dataset.head()


# Daily data has the following fields. Thanks to the people who prepared it because it is very well processed data with even scaled features. 
# 
# instant: Record index
# 
# dteday: Date
# 
# season: Season (1:springer, 2:summer, 3:fall, 4:winter)
# 
# yr: Year (0: 2011, 1:2012)
# 
# mnth: Month (1 to 12)
# 
# holiday: weather day is holiday or not (extracted from Holiday Schedule)
# 
# weekday: Day of the week
# 
# workingday: If day is neither weekend nor holiday is 1, otherwise is 0.
# 
# weathersit: (extracted from Freemeteo)
# 1: Clear, Few clouds, Partly cloudy, Partly cloudy
# 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
# 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
# 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# 
# temp: Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
# 
# atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
# 
# hum: Normalized humidity. The values are divided to 100 (max)
# 
# windspeed: Normalized wind speed. The values are divided to 67 (max)
# 
# casual: count of casual users
# 
# registered: count of registered users
# 
# cnt: count of total rental bikes including both casual and registered

# #### Plot of 2 years number of sharing (cnt)

# In[4]:


plt.figure(figsize=(15,10))
plt.plot(dataset['Close'], color='blue')
plt.show()


# #### Data exploration and Manipulation

# * Number of bike sharing is 22 only at 2012-10-29 and such a low value deserves a special attention.
# 
# * There was a hurricane at Washington at that day.
# 
# * Since it is such an extraordinary day, hurricane and the following days data will be replaced by the average of that month.

# In[5]:


#temp = dataset[dataset.yr == 1]
#temp = temp[temp.mnth == 10]
#print(temp.cnt.mean())


# In[6]:


#temp.head()


# In[7]:


#print(dataset['cnt'][667], dataset['cnt'][668])


# In[8]:


#dataset['cnt'][667] = 6414
#dataset['cnt'][668] = 6414


# #### One Hot Encoding

# We should apply one hot encoding for categorical features. In our case weekday, weathersit and mnth features are one hot encoded.

# In[9]:


# In[10]:



# In[11]:




# #### Scaling

# Thanks to the guys prepared the original data, they scaled all features. That is why we have to apply it only for our value Y which is cnt. It is also a discussion whether Y value should be scaled or not in sucha model but we did. 

# In[12]:

#Seleciona o tipo de escalonamento
scaler = MinMaxScaler(feature_range=(0, 1))

#Escalonamento de dados 
scaled = scaler.fit_transform(array(dataset['Close']).reshape(len(dataset['Close']), 1))
dt_train = dataset.iloc[:,1:13]
freature_columns = dataset.iloc[:,1:13].columns
dataset = scaler.fit_transform(dt_train)
dataset = pd.DataFrame(data=dataset,columns=freature_columns)
#Junta a coluna com os dados
series = pd.DataFrame(scaled)
series.columns = ['cntscl']



# In[13]:


dataset = pd.merge(dataset, series, left_index=True, right_index=True)
columns = dataset.columns

# In[14]:


dataset.head()


# #### Data Splitting

# In[15]:


number_of_test_data = 100
number_of_holdout_data = 100
number_of_training_data = len(dataset) - number_of_holdout_data - number_of_test_data
print ("total, train, test, holdout:", len(dataset), number_of_training_data, number_of_test_data, number_of_holdout_data)


# In[16]:


datatrain = dataset[:number_of_training_data]
datatest = dataset[-(number_of_test_data+number_of_holdout_data):-number_of_holdout_data]
datahold = dataset[-number_of_holdout_data:]


# ### Preparing 3-Dimensional Input for Sequential Model

# The following steps show the way how to prepare input for a sequential model by using TimeSeriesGenerator.

# In[17]:


in_seq1 = array(datatrain['Day'])
in_seq2 = array(datatrain['Month']) 
in_seq3 = array(datatrain['Volume'])
in_seq4 = array(datatrain['Difference'])
out_seq_train = array(datatrain['cntscl'])


# In[18]:

in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.reshape((len(in_seq4), 1))
out_seq_train = out_seq_train.reshape((len(out_seq_train), 1))


# In[19]:


datatrain_feed = hstack((in_seq1, in_seq2, in_seq3,in_seq4,out_seq_train))


# In[20]:



in_seq1 = array(datatest['Day'])
in_seq2 = array(datatest['Month']) 
in_seq3 = array(datatest['Volume'])
in_seq4 = array(datatest['Difference'])
out_seq_test = array(datatest['cntscl'])


# In[21]:


in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.reshape((len(in_seq4), 1))
out_seq_test = out_seq_test.reshape((len(out_seq_test), 1))


# In[22]:


datatest_feed = hstack((in_seq1, in_seq2, in_seq3,in_seq4,out_seq_test))


# In[23]:



in_seq1 = array(datahold['Day'])
in_seq2 = array(datahold['Month']) 
in_seq3 = array(datahold['Volume'])
in_seq3 = array(datahold['Difference'])
out_seq_hold = array(datahold['cntscl'])


# In[24]:


in_seq1 = in_seq1.reshape((len(in_seq1), 1))
in_seq2 = in_seq2.reshape((len(in_seq2), 1))
in_seq3 = in_seq3.reshape((len(in_seq3), 1))
in_seq4 = in_seq4.reshape((len(in_seq4), 1))
out_seq_hold = out_seq_hold.reshape((len(out_seq_hold), 1))


# In[25]:


datahold_feed = hstack((in_seq1, in_seq2, in_seq3,in_seq4,out_seq_hold))


# In[26]:


n_features = datatrain_feed.shape[1]
n_input = 5
generator_train = TimeseriesGenerator(datatrain_feed, out_seq_train, length=n_input, batch_size=len(datatrain_feed))


# In[27]:


for i in range(len(generator_train)):
	x, y = generator_train[i]
	print('%s => %s' % (x, y))


# In[28]:


generator_test = TimeseriesGenerator(datatest_feed, out_seq_test, length=n_input, batch_size=1)


# In[29]:


for i in range(len(generator_test)):
	x, y = generator_test[i]
	print('%s => %s' % (x, y))


# In[30]:


generator_hold = TimeseriesGenerator(datahold_feed, out_seq_hold, length=n_input, batch_size=1)


# In[31]:


for i in range(len(generator_hold)):
	x, y = generator_hold[i]
	print('%s => %s' % (x, y))


# ### Modelling and Training

# We have created a small RNN with 4 nodes. 
# Number of total parameters in the model is 93. 
# Number of timesteps in one batch is 10. 
# Activation function is relu both for RNN and Output layer.
# Optimizer is adam.
# Loss function is mean squared error.
# Learning rate is 0.0001.
# Number of epocs is 3,000.

# #### Creating the SimpleRNN Model

# In[32]:


print("timesteps, features:", n_input, n_features)


# In[54]:


model = Sequential()

model.add(SimpleRNN(5, activation='relu', input_shape=(n_input, n_features), return_sequences = False))
model.add(Dense(1, activation='relu'))

adam = Adam(lr=0.0001)
model.compile(optimizer=adam, loss='mse')


# In[55]:


model.summary()


# #### Training the Model

# In[56]:


score = model.fit_generator(generator_train, epochs=30, verbose=2, validation_data=generator_test)


# #### Plot of Training and Test Loss Functions

# In[39]:


losses = score.history['loss']
val_losses = score.history['val_loss']
plt.figure(figsize=(10,5))
plt.plot(losses, label="Treino")
plt.plot(val_losses, label="Teste")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.show()


# ### Predictions for Test Data

# #### Predicting for Test Data

# In[40]:


df_result = pd.DataFrame({'Actual' : [], 'Prediction' : []})

for i in range(len(generator_test)):
    x_test, y_test = generator_test[i]
    x_input = array(x_test).reshape((1, n_input, n_features))
    yhat = model.predict(x_input, verbose=2)
    
    trainPredict_dataset_like = np.zeros(shape=(len(y_test),12))
    trainPredict_dataset_like[:,0] = y_test[:,0]
    y_test = scaler.inverse_transform(trainPredict_dataset_like)[:,0]
    
    trainPredict_dataset_like = np.zeros(shape=(len(yhat),12))
    trainPredict_dataset_like[:,0] = yhat[:,0]
    yhat = scaler.inverse_transform(trainPredict_dataset_like)[:,0]
    
    df_result = df_result.append({'Actual': y, 'Prediction': yhat}, ignore_index=True)


# #### Tabulating Actuals, Predictions and Differences

# In[41]:


df_result['Diff'] = 100 * (df_result['Prediction'] - df_result['Actual']) / df_result['Actual']


# In[42]:


df_result


# #### Calculating the Correctness for Test Data

# In[43]:


mean = df_result['Actual'].mean()
mae = (df_result['Actual'] - df_result['Prediction']).abs().mean()

print("===============================================")
print("Média de teste: ", mean)
print("Diferença:", mae)
print("Diferença/Média Percentual: ", 100*mae/mean,"%")
print("Correção: ", 100 - 100*mae/mean,"%")
print("===============================================")


# #### Plot of Actuals and Predictions for Test Data

# In[44]:


plt.figure(figsize=(15,10))
plt.plot(df_result['Actual'], color='blue')
plt.plot(df_result['Prediction'], color='red')
plt.title("Gráfico de teste")
plt.show()


# ### Predictions for Hold-Out Data

# #### Predicting for Hold-Out Data

# In[45]:


df_result = pd.DataFrame({'Actual' : [], 'Prediction' : []})

for i in range(len(generator_hold)):
    x, y = generator_hold[i]
    x_input = array(x).reshape((1, n_input, n_features))
    yhat = model.predict(x_input, verbose=2)
    
    trainPredict_dataset_like = np.zeros(shape=(len(y),12))
    trainPredict_dataset_like[:,0] = y[:,0]
    y = scaler.inverse_transform(trainPredict_dataset_like)[:,0]
    
    trainPredict_dataset_like = np.zeros(shape=(len(yhat),12))
    trainPredict_dataset_like[:,0] = yhat[:,0]
    yhat = scaler.inverse_transform(trainPredict_dataset_like)[:,0]
    
    df_result = df_result.append({'Actual': y, 'Prediction': yhat}, ignore_index=True)


# #### Tabulating Actuals, Predictions and Differences for Hold-Out Data

# In[46]:


df_result['Diff'] = 100 * (df_result['Prediction'] - df_result['Actual']) / df_result['Actual']


# In[47]:


df_result


# #### Calculating the Correctness for Hold-Out Data

# In[48]:


mean = df_result['Actual'].mean()
mae = (df_result['Actual'] - df_result['Prediction']).abs().mean()

print("Média : ", mean)
print("Diferença:", mae)
print("Diferença/Média Percentual: ", 100*mae/mean,"%")
print("Nível correção: ", 100 - 100*mae/mean,"%")
print("===============================================")


# #### Plot of Actuals and Predictions for Hold-Out Data

# In[49]:

graph_test = pd.DataFrame()

diff = []
for i in range(len(df_result)):
     i_old = int(i)-1
     difference_num = (df_result.iloc[i_old,1] - df_result.iloc[i,1])
     
     if (difference_num > 0):
         diff.append(0)
     else:
         diff.append(1)


graph_test['Prediction'] = df_result['Prediction']
graph_test['Difference_Afirm'] = diff
plt.figure(figsize=(15,10))
#plt.plot(df_result['Actual'], color='blue')
#plt.plot(graph_test['Prediction'], color='red')
sns.set(style='darkgrid')
ax = sns.lineplot(x=graph_test.index,y=graph_test['Prediction'])
ax.set(xlabel='Indices')
labels = graph_test['Difference_Afirm'].dropna().unique().tolist()

for label in labels:
    sns.lineplot(x=graph_test[graph_test['Difference_Afirm'] == 1].index,
                 y=graph_test[graph_test['Difference_Afirm'] == 1]['Prediction'],
                 color='green')
    ax.axvspan(graph_test[graph_test['Difference_Afirm'] == 1].index[0],
               graph_test[graph_test['Difference_Afirm'] == 1].index[-1],
               alpha=0.2,
               color='green')
    
    sns.lineplot(x=graph_test[graph_test['Difference_Afirm'] == 0].index,
                 y=graph_test[graph_test['Difference_Afirm'] == 0]['Prediction'],
                 color='red')
    ax.axvspan(graph_test[graph_test['Difference_Afirm'] == 0].index[0],
               graph_test[graph_test['Difference_Afirm'] == 0].index[-1],
               alpha=0.2,
               color='red')

plt.title("Gráfico de correção")
plt.show()


# In[ ]:




