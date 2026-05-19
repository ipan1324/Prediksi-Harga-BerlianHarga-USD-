from flask import Flask
from flask import render_template
from flask import request

import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('models/linear_regression_model.pkl')
scaler = joblib.load('models/scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

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

    data = np.array([[carat, cut, color, clarity, depth, table, x, y, z]])

    scaled = scaler.transform(data)

    prediction = model.predict(scaled)[0]

    return render_template(
        'result.html',
        prediction=round(prediction, 2)
    )

if __name__ == '__main__':
    app.run(debug=True)