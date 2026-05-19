import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# load dataset

df = pd.read_csv('data/diamonds.csv')

# handling missing values

df.dropna(inplace=True)

# encoding categorical columns

categorical_columns = ['cut', 'color', 'clarity']

encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# features and target

X = df.drop('price', axis=1)
y = df['price']

# scaling

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# save scaler

joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(encoders, 'models/encoder.pkl')

# split dataset

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled,
    y,
    test_size=0.3,
    random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    random_state=42
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

pred_lr = lr_model.predict(X_test)

mae_lr = mean_absolute_error(y_test, pred_lr)
rmse_lr = np.sqrt(mean_squared_error(y_test, pred_lr))
r2_lr = r2_score(y_test, pred_lr)

print('LINEAR REGRESSION')
print('MAE:', mae_lr)
print('RMSE:', rmse_lr)
print('R2:', r2_lr)

joblib.dump(lr_model, 'models/linear_regression.pkl')

import matplotlib.pyplot as plt

residuals = y_test - pred_lr

plt.figure(figsize=(8,6))
plt.scatter(pred_lr, residuals)
plt.axhline(y=0)
plt.xlabel('Predicted')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.savefig('visualizations/residual_plot.png')

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

ann_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

history_ann = ann_model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    verbose=1
)

pred_ann = ann_model.predict(X_test)

mae_ann = mean_absolute_error(y_test, pred_ann)
rmse_ann = np.sqrt(mean_squared_error(y_test, pred_ann))

print('ANN')
print('MAE:', mae_ann)
print('RMSE:', rmse_ann)

ann_model.save('models/ann_model.h5')

plt.figure(figsize=(8,6))
plt.plot(history_ann.history['loss'])
plt.plot(history_ann.history['val_loss'])
plt.title('ANN Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])
plt.savefig('visualizations/loss_curve.png')

from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_val_lstm = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

lstm_model = Sequential([
    LSTM(64, input_shape=(1, X_train.shape[1])),
    Dense(32, activation='relu'),
    Dense(1)
])

lstm_model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

history_lstm = lstm_model.fit(
    X_train_lstm,
    y_train,
    validation_data=(X_val_lstm, y_val),
    epochs=30,
    batch_size=32,
    verbose=1
)

pred_lstm = lstm_model.predict(X_test_lstm)

mae_lstm = mean_absolute_error(y_test, pred_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test, pred_lstm))

print('LSTM')
print('MAE:', mae_lstm)
print('RMSE:', rmse_lstm)

lstm_model.save('models/lstm_model.h5')

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

inertia_values = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8,6))
plt.plot(range(1,11), inertia_values, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.savefig('visualizations/elbow_method.png')

kmeans_model = KMeans(n_clusters=3, random_state=42)
kmeans_model.fit(X_scaled)

silhouette = silhouette_score(X_scaled, kmeans_model.labels_)

print('KMEANS')
print('Inertia:', kmeans_model.inertia_)
print('Silhouette Score:', silhouette)

joblib.dump(kmeans_model, 'models/kmeans_model.pkl')

import numpy as np

class SimpleBackpropagation:

    def __init__(self, input_size, hidden_size, output_size):

        self.W1 = np.random.randn(input_size, hidden_size)
        self.W2 = np.random.randn(hidden_size, output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):

        self.z1 = np.dot(X, self.W1)
        self.a1 = self.sigmoid(self.z1)

        self.z2 = np.dot(self.a1, self.W2)
        self.output = self.sigmoid(self.z2)

        return self.output

    def backward(self, X, y, learning_rate):

        error = y - self.output

        d_output = error * self.sigmoid_derivative(self.output)

        error_hidden = d_output.dot(self.W2.T)

        d_hidden = error_hidden * self.sigmoid_derivative(self.a1)

        self.W2 += self.a1.T.dot(d_output) * learning_rate
        self.W1 += X.T.dot(d_hidden) * learning_rate

    def train(self, X, y, epochs=1000, learning_rate=0.01):

        for epoch in range(epochs):

            output = self.forward(X)

            self.backward(X, y, learning_rate)

            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f'Epoch {epoch}, Loss: {loss}')

models = ['Linear Regression', 'ANN', 'LSTM']
mae_scores = [mae_lr, mae_ann, mae_lstm]

plt.figure(figsize=(8,6))
plt.bar(models, mae_scores)
plt.ylabel('MAE')
plt.title('Model Comparison')
plt.savefig('visualizations/model_comparison.png')