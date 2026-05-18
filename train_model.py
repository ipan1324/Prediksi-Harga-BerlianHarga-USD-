import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

print("Loading and preprocessing data...")
df = pd.read_csv("data/diamonds.csv")

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

# Encoding
encoder_cut = LabelEncoder()
encoder_color = LabelEncoder()
encoder_clarity = LabelEncoder()

df["cut"] = encoder_cut.fit_transform(df["cut"])
df["color"] = encoder_color.fit_transform(df["color"])
df["clarity"] = encoder_clarity.fit_transform(df["clarity"])

# Features & Target
X = df.drop("price", axis=1)
y = df["price"]

# Data Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler so app.py can use it
joblib.dump(scaler, "models/scaler.pkl")

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42
)

# 1. Linear Regression
print("\n--- Training Linear Regression ---")
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print(f"R2 Score: {r2_score(y_test, pred_lr):.4f}")
joblib.dump(lr, "models/linear_regression_model.pkl")

# 2. Artificial Neural Network (MLP)
print("\n--- Training ANN (MLPRegressor) ---")
# Using the same architecture as the original keras notebook (64, 32)
ann = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
ann.fit(X_train, y_train)
pred_ann = ann.predict(X_test)
print(f"R2 Score: {r2_score(y_test, pred_ann):.4f}")
joblib.dump(ann, "models/ann_model.pkl")

# 3. Random Forest
print("\n--- Training Random Forest ---")
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
print(f"R2 Score: {r2_score(y_test, pred_rf):.4f}")
joblib.dump(rf, "models/random_forest_model.pkl")

print("\nAll models successfully trained and saved!")
