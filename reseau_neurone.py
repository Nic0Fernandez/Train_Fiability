import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from traitement_données import traitement_donnees
import tensorflow as tf
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from traitement_données import traitement_donnees
from sklearn.preprocessing import StandardScaler


# Définir la métrique R² personnalisée
def r2_score_metric(y_true, y_pred):
    SS_res = tf.reduce_sum(tf.square(y_true - y_pred))
    SS_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return (1 - SS_res / (SS_tot + tf.keras.backend.epsilon()))


data = traitement_donnees()

anomaly_columns = ['Feed Volume (m3)', 'Brine Volume (m3)', 'Product Volume (m3)','Recovery (%)', 'Normalized DP (bars)']

data = data.dropna()

X = data.loc[:,data.columns != 'Normalized DP (bars)'] #'Feed Volume (m3)','Brine Volume (m3)','Product Volume (m3)','Recovery (%)','Normalized DP (bars)'
Y = data.loc[:,'Normalized DP (bars)']


# Normalisation des données
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_Y = StandardScaler()
Y_scaled = scaler_Y.fit_transform(Y.values.reshape(-1, 1))

# Définir la fonction pour créer des séquences
def create_sequences(X, Y, time_steps=10):
    Xs, Ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:i+time_steps])
        Ys.append(Y[i+time_steps])
    return np.array(Xs), np.array(Ys)

time_steps = 10
X_seq, Y_seq = create_sequences(X_scaled, Y_scaled, time_steps)

# Séparation des données en ensemble d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(X_seq, Y_seq, test_size=0.3, random_state=42)

# Définition du modèle LSTM
model = keras.Sequential([
    keras.layers.LSTM(100, activation='relu', return_sequences=True, input_shape=(time_steps, X_train.shape[2])),
    keras.layers.Dropout(0.2),
    keras.layers.LSTM(50, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)   
])

model.compile(optimizer='adam', loss='mean_squared_error',metrics=[tf.keras.metrics.RootMeanSquaredError(), r2_score_metric])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Affichage du résumé du modèle
model.summary()

# Entraînement du modèle
history = model.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

# Évaluation du modèle
results = model.evaluate(X_test, Y_test)
loss = results[0]
rmse = results[1]
r2 = results[2]
print(f"Test Loss: {loss}")
print(f"Test RMSE: {rmse}")
print(f"Test R² Score: {r2}")

# Prédictions
Y_pred_scaled = model.predict(X_test)

# Inverse scaling
Y_test_inv = scaler_Y.inverse_transform(Y_test)
Y_pred_inv = scaler_Y.inverse_transform(Y_pred_scaled)

# Visualisation des résultats
plt.figure(figsize=(10, 6))
plt.scatter(Y_test_inv, Y_pred_inv)
plt.xlabel('Valeurs réelles (Y_test)')
plt.ylabel('Valeurs prédites (Y_pred)')
plt.grid(True)
plt.show()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['root_mean_squared_error'], label='RMSE')
plt.plot(history.history['val_root_mean_squared_error'], label='val_RMSE')
plt.xlabel('Epoch')
plt.ylabel('RMSE')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(history.history['r2_score_metric'], label='R²')
plt.plot(history.history['val_r2_score_metric'], label='val_R²')
plt.xlabel('Epoch')
plt.ylabel('R²')
plt.legend()
plt.grid(True)
plt.show()