import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# !!! 
# pentru pentru a putea rula programul trebuie 
# → file open folder → folderul care contine si codu si fisierul excel

file = 'dataset_lab3.xlsx'

# luam datele din fisierul .xlsx in care avem datele modificate pentru problema noastra
data_frame = pd.read_excel(file)

# am transformat data frameul in matrice
raw_data = data_frame.to_numpy()

# etichetele sunt ultima coloana din matrice, deci le stocam in y
y = [row[-1] for row in raw_data]

# caracteristicile sumt in restul matricii raw_data
X = [row[:-1] for row in raw_data]

# matricea X are liniile salvate ca liste, le transformam in vectori
X = [np.array(row) for row in X]

# Impartim datele in date de test si de testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Putem alege Gauss deoarece avem 2 etichete/ clase
clf = GaussianNB()

# Cream modelul si pentru regresia logistica
model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)

# Antrenam clasificatorul Bayes Naive
clf.fit(X_train, y_train)

# Antenam modelul cu regresie logistica
model.fit(X_train, y_train)

# Testam clasificatorul cu valorile de test
predictions = clf.predict(X_test)

# Testam modelul cu regresia logistica
y_pred = model.predict(X_test)

# Calculam precizia clasificatorului BN
accuracyBN = accuracy_score(y_test, predictions)

# Calculam precizia clasificatorului cu regresia Logistica
accuracyLR = accuracy_score(y_test, y_pred)

# Afisam  acuratetea ambelor modele
print("Accuracy Bayes Naive :", accuracyBN)
print("Accuracy Logistical Regression :", accuracyLR)