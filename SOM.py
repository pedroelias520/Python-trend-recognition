import sys
from minisom import MiniSom
import time
import pandas as pd
from sklearn.datasets import load_breast_cancer
import semantic_version

som_grid_rows = 5194
som_grid_column = 7
interations = 500
sigma = 1
learning_rate = 0.5

data = pd.read_csv('VALE3.SA.csv')

som = MiniSom(x=3, y=3, input_len=7,sigma=sigma,learning_rate=learning_rate,neighborhood_function='bubble')
som.random_weights_init(data)

start_time = time.time()
som.train_random(data,interations) #treino com 500 interações
elapsed_time = time.time() - start_time
print(elapsed_time," seconds")

print("Quantificação...")
qnt = som.quantization(data)

print(qnt)