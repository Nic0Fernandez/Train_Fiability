import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

columns = ['weeks','Normalized DP (bars)']

def traitement_donnees():
    df = pd.read_excel('Dataset-D.xlsx', sheet_name="Train 2")
    df = df.loc[:,~df.columns.str.contains('^Unnamed')]
    df = df.drop(columns=['Day','K','Date','Train Status','Train Status Code','Feed Volume (m3)','Brine Volume (m3)','Product Volume (m3)','Recovery (%)'])
    df = df.dropna(subset=['Normalized DP (bars)'])
    return df

data = traitement_donnees()
print(data.corr())

data300 = data[:][:300]

#sns.lmplot(x="weeks", y="Normalized DP (bars)", data=data300, order=2, ci=None)
#plt.show()

#plt.figure(figsize=(10, 8))
#sns.heatmap(data.corr(), annot=True)
#plt.show()



