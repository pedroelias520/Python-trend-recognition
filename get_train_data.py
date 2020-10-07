import matplotlib.pyplot as plt
import seaborn as sns
import trendet
import numpy as np

#Pegar os arrays disponíveis
sns.set(style='darkgrid')
company = 'REP'
df = trendet.identify_all_trends(company, 'Spain', '01/01/2018', '01/01/2019',window_size=5,identify='both')
df.reset_index(inplace=True)
df = df[['Date','Open','Close','Up Trend','Down Trend']]

#Transforma os dados em binários 
df['Up Trend'] = df['Up Trend'].replace(['A','B','C','D','E','F'],1)
df['Down Trend'] = df['Down Trend'].replace(['A','B','C','D','E','F'],1)

df['Up Trend'] = df['Up Trend'].replace(np.nan,0)
df['Down Trend'] = df['Down Trend'].replace([np.nan],0)

#Calcula a diferença presente no dataframe
difference = []
for index, row in df.iterrows():
    difference.append(row['Open']-row['Close'])
    
x = []
for num in difference:
    x.append('{0:.3g}'.format(num))      

df['Difference'] = x #Adiciona o array com as diferenças no dataframe
df.to_csv(company+'_Train.csv') #Transforma o dataframe em .csv

#Faz um csv para teste
test_df = df
del test_df['Up Trend']
del test_df['Down Trend']
test_df.to_csv(company+'_Test.csv')

