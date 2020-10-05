from yahooquery import Ticker
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as gx
from tkinter import *

tickers_ibov = {'ABEV3.SA', 'AZUL4.SA', 'B3SA3.SA', 'BBAS3.SA', 'BBDC4.SA', 'BBSE3.SA', 'BPAC11.SA', 'BRAP4.SA',
                'BRDT3.SA', 'BRFS3.SA', 'BRKM5.SA', 'BRML3.SA', 'BTOW3.SA', 'CCRO3.SA', 'CIEL3.SA', 'CMIG4.SA',
                'COGN3.SA', 'CRFB3.SA', 'CSAN3.SA', 'CSNA3.SA', 'CVCB3.SA', 'CYRE3.SA', 'ECOR3.SA', 'EGIE3.SA',
                'ELET6.SA', 'EMBR3.SA', 'ENBR3.SA', 'EQTL3.SA', 'FLRY3.SA', 'GGBR4.SA', 'GNDI3.SA', 'GOAU4.SA',
                'GOLL4.SA', 'HAPV3.SA', 'HGTX3.SA', 'HYPE3.SA', 'IGTA3.SA', 'IRBR3.SA', 'ITSA4.SA', 'ITUB4.SA ',
                'JBSS3.SA', 'KLBN11.SA ', 'LAME4.SA ', 'LREN3.SA', 'MGLU3.SA', 'MRFG3.SA', 'MRVE3.SA', 'MULT3.SA',
                'NTCO3.SA', 'PCAR4.SA', 'PETR4.SA', 'QUAL3.SA', 'RADL3.SA', 'RAIL3.SA', 'RENT3.SA', 'SANB11.SA',
                'SBSP3.SA', 'SMLS3.SA', 'SULA11.SA', 'SUZB3.SA', 'TAEE11.SA', 'TIMP3.SA', 'TOTS3.SA', 'UGPA3.SA',
                'USIM5.SA', 'VALE3.SA', 'VIVT4.SA', 'VVAR3.SA', 'WEGE3.SA', 'YDUQ3.SA'}
times = {'max', '5', '10', '15', '30'}
label_text = ""

class Interface():
    def __init__(self):
        self.company = ""
        self.time = ""
        self.root = Tk()
        self.root.geometry('700x300')
        self.root.resizable(False, False)
        self.root.title("Bem-vindo")

        # Dropdown
        mainframe = Frame(self.root)
        mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
        mainframe.columnconfigure(1, weight=1)
        mainframe.rowconfigure(0, weight=1)
        mainframe.pack(pady=100, padx=100)

        # Dicionário de e companias
        tkvar = StringVar(self.root)
        tkvar.set('Empresas')
        time = StringVar(self.root)
        time.set('Tempo de exibição')

        # Cria popupMenu
        popupMenu = OptionMenu(mainframe, tkvar, *tickers_ibov)
        popupMenu.grid(row=0, column=0)
        timepopup = OptionMenu(mainframe, time, *times)
        timepopup.grid(row=0, column=1)

        Button(mainframe, text='Gerar gráfico', command=self.GenerateGraph, width=12,height=1).grid(row=0, column=3)

        def change_dropdown(*args):
            self.company = tkvar.get()
            print(self.company)

        def time_dropdown(*args):
            self.time = time.get()
            print(self.time)

        time.trace('w', time_dropdown)
        tkvar.trace('w', change_dropdown)
        self.root.mainloop()

    def GenerateGraph(self):
        if(self.company and self.time):
            self.csv_generation(self.company, self.time)
            self.csv_to_graph(self.company)
        else:
            print("Todos campos devem ser selecionados")
            label_text = "Todos os campos devem ser selecionados"


    def csv_generation(self,company_name, period_time):
        dados_yf = yf.download(tickers=company_name, period= period_time)
        path = 'C:/Users/Pedro Elias/Documents/Projetos_GitHub/Python-trend-recognition/'
        dados_yf.to_csv(path + company_name + ".csv")
        dados_yf.to_excel(path+company_name+'.xlsx')

    def csv_to_graph(self,csv_company):
        data_frame = pd.read_csv(csv_company + ".csv")
        graph_fig = go.Figure(data=[go.Candlestick(x=data_frame['Date'],
                                                   open=data_frame['Open'],
                                                   high=data_frame['High'],
                                                   low=data_frame['Low'],
                                                   close=data_frame['Close']
                                                   )])
        #graph_fig.add_trace(go.Scatter(x=data_frame['Date'],y=data_frame['Close'],name='Medium',line=dict(color='purple',width=4)))
        graph_fig.show()


Interface()
