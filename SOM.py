import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stock_train_data = pd.DataFrame(pd.read_csv("BBVA_Train.csv"))
stock_test_data = pd.DataFrame(pd.read_csv("REP_Test.csv"))

# treinamento de classificadores
# Onde o segmento de treino achará os passageiros sobreviventes


titanic_dataset_train_Column = stock_train_data[["Difference"]]
#titanic_train_segment = (stock_train_data[["Difference"]] == 1) #A Diferença tem que ser maior ou menor que zero 
titanic_train_segment = []
for row in titanic_dataset_train_Column.values:
    if (row < 0):
        titanic_train_segment.append(True)
    else: 
        titanic_train_segment.append(False)

segment = np.array(titanic_train_segment).ravel()

test_column = stock_test_data[["Difference"]]


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(titanic_dataset_train_Column, segment)
some_person = test_column.iloc[[1]]

# teste de classificador
sgd_clf.predict(some_person)

# validação cruzada
from sklearn.model_selection import cross_val_score
cross_val_score(sgd_clf, titanic_dataset_train_Column, segment, cv=3, scoring="accuracy")

# matriz de consfusão
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, titanic_dataset_train_Column, segment, cv=3)
confusion_matrix(segment, y_train_pred)

# controle de precisão
from sklearn.metrics import precision_score, recall_score
print(precision_score(segment, y_train_pred))

