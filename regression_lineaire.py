from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from traitement_données import traitement_donnees
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importer panda dataframe des données traitées depuis traitement_données.py 

data = traitement_donnees()

# Diviser les données en données d'entraînement et de test
X = data.loc[:,data.columns != 'Normalized DP (bars)']
Y = data.loc[:,'Normalized DP (bars)']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Créer un modèle de régression linéaire 
model = LinearRegression()
model.fit(X_train,Y_train)

# Prédire les valeurs de test
Y_pred = model.predict(X_test)

#Évaluation du modèle
score = r2_score(Y_test,Y_pred)
rmse = mean_squared_error(Y_test, Y_pred, squared=False)

print("R2:", score)
print("RMSE:", rmse)

# Afficher les résultats 
plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', color=('r'))
plt.xlabel('Valeurs réelles (Y_test)')
plt.ylabel('Valeurs prédites (Y_pred)')
plt.grid(True)
plt.show()