import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import joblib

df = pd.read_csv("data/diamonds.csv")

if "Unnamed: 0" in df.columns:
    df.drop("Unnamed: 0", axis=1, inplace=True)

encoder_cut = LabelEncoder()
encoder_color = LabelEncoder()
encoder_clarity = LabelEncoder()

df["cut"] = encoder_cut.fit_transform(df["cut"])
df["color"] = encoder_color.fit_transform(df["color"])
df["clarity"] = encoder_clarity.fit_transform(df["clarity"])

print("Cut mappings:", dict(zip(encoder_cut.classes_, encoder_cut.transform(encoder_cut.classes_))))
print("Color mappings:", dict(zip(encoder_color.classes_, encoder_color.transform(encoder_color.classes_))))
print("Clarity mappings:", dict(zip(encoder_clarity.classes_, encoder_clarity.transform(encoder_clarity.classes_))))

X = df.drop("price", axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "models/linear_regression_model.pkl")
print("Model saved to models/linear_regression_model.pkl")
