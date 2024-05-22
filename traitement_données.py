import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

columns = ['weeks','Feed Volume (m3)','Brine Volume (m3)','Product Volume (m3)','Recovery (%)','Normalized DP (bars)']

def traitement_donnees():
    df_list=[]
    for i in range(1,15):
        if i == 12:
            continue
        else:
            df = pd.read_excel('Dataset-D.xlsx', sheet_name="Train " + str(i))
            df = df.loc[:,~df.columns.str.contains('^Unnamed')]
            df = df.drop(columns=['Day','K','Date','Train Status','Train Status Code'])
            df = df.iloc[:1624]
            df_list.append(df)
    giant_df = pd.concat(df_list, ignore_index=True)
    return giant_df


data = traitement_donnees()

data300 = data[:][:300]

#sns.lmplot(x="weeks", y="Normalized DP (bars)", data=data300, order=2, ci=None)
#plt.show()

#plt.figure(figsize=(10, 8))
#sns.heatmap(data.corr(), annot=True)
#plt.show()



