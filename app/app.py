from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("../models/linear_regression_model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    carat = float(request.form['carat'])
    cut = float(request.form['cut'])
    color = float(request.form['color'])
    clarity = float(request.form['clarity'])
    depth = float(request.form['depth'])
    table = float(request.form['table'])
    x = float(request.form['x'])
    y = float(request.form['y'])
    z = float(request.form['z'])

    import pandas as pd
    data = pd.DataFrame([[
        carat,
        cut,
        color,
        clarity,
        depth,
        table,
        x,
        y,
        z
    ]], columns=['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x', 'y', 'z'])

    prediction = model.predict(data)

    return render_template(
        "index.html",
        prediction_text=f'Prediksi Harga Berlian: ${prediction[0]:,.2f}'
    )

if __name__ == "__main__":
    app.run(debug=True)