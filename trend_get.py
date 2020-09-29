from yahooquery import Ticker
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from tkinter import *

def __init__(self):
    self.company= ""
    self.root  = Tk()
    user_name = "%USERNAME%"
    self.root.resizable(True,True)
    self.root.title("Bem-vindo"+user_name)

    Button(self.root,text='Selecionar Empresa').grid(row=0, column=0,pady=0)
    self.root.mainloop()



def csv_generation (company_name):
    dados_yf = yf.download(tickers=company_name, period='max')
    path = 'C:/Users/Pedro Elias/Documents/Projetos_GitHub/Python-trend-recognition/'
    dados_yf.to_csv(path + company_name+".csv")

def csv_to_graph (csv_company):
    data_frame = pd.read_csv(csv_company)
    graph_fig = go.Figure(data=[go.Candlestick(x=data_frame['Date'],
                                               open=data_frame['Open'],
                                               high=data_frame['High'],
                                               low=data_frame['Low'],
                                               close=data_frame['Close']
                                               )])
    graph_fig.show()