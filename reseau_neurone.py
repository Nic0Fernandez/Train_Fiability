import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from traitement_données import traitement_donnees
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from traitement_données import traitement_donnees
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


data = traitement_donnees()

X = data.loc[:,data.columns != 'Normalized DP (bars)']
Y = data.loc[:,'Normalized DP (bars)']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = keras.Sequential(
    [
        keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1)
    ]
)

model.compile(optimizer='adam',loss='MeanSquaredError')
model.summary()
history = model.fit(X_train, Y_train, epochs=100, validation_split=0.2)
Y_pred = model.predict(X_test)

score = r2_score(Y_test,Y_pred)
rmse = mean_squared_error(Y_test, Y_pred, squared=False)

print("R2:", score)
print("RMSE:", rmse)

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred)
plt.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', color=('r'))
plt.xlabel('Valeurs réelles (Y_test)')
plt.ylabel('Valeurs prédites (Y_pred)')
plt.grid(True)
plt.show()


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.show()

