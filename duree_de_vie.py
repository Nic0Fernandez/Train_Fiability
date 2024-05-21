import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import numpy as np
from traitement_données import traitement_donnees

df = pd.read_excel('Dataset-D.xlsx', sheet_name="Train 2")
df = df.loc[:,~df.columns.str.contains('^Unnamed')]
df = df.drop(columns=['Day','K','Date','Train Status','Train Status Code','Feed Volume (m3)','Brine Volume (m3)','Product Volume (m3)','Recovery (%)'])

durations = df["weeks"]
event = df["Normalized DP (bars)"]
for i in range(len(event)):
    if df["Normalized DP (bars)"][i] > 0:
        event[i] = 0
    elif np.isnan(df["Normalized DP (bars)"][i]):
        event[i] = 1

event_observed = event

kmf = KaplanMeierFitter()
kmf.fit(durations, event_observed)

kmf.plot_survival_function()
plt.title('Kaplan-Meier Estimator')
plt.xlabel('Weeks')
plt.ylabel('Survival Probability')
plt.show()


def MRL(t):
    r_t = kmf.predict(t)
    somme = []
    for i in range (len(event)-t):
        somme.append(kmf.predict(t+i))
    mrl_t = np.sum(somme)/r_t
    return mrl_t


time_points = [0, 100, 200, 300]
#survival_probabilities = kmf.survival_function_at_times(time_points)
#print(survival_probabilities)
for i in range (len(time_points)):
    print(MRL(time_points[i]))