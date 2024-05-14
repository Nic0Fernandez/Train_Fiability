import pandas as pd
import openpyxl as op


data = []

# Importation des données
for i in range(1,15):
    data.append(pd.read_excel("Projet/Dataset-D.xlsx", sheet_name="Train "+str(i)))

# Traitement des données (suppression des lignes où Normalized DP n'existe pas)
for i in range(14):
    data[i] = data[i].dropna(subset=['Normalized DP (bars)'])

# Affichage des données
print(data[0])
print(data[1])






