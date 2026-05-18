from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load Scaler
scaler = joblib.load("../models/scaler.pkl")

# Load Models
models = {
    'linear_regression': joblib.load("../models/linear_regression_model.pkl"),
    'ann': joblib.load("../models/ann_model.pkl"),
    'random_forest': joblib.load("../models/random_forest_model.pkl")
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_type = request.form['model_type']
        
        carat = float(request.form['carat'])
        cut = float(request.form['cut'])
        color = float(request.form['color'])
        clarity = float(request.form['clarity'])
        depth = float(request.form['depth'])
        table = float(request.form['table'])
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])

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

        # Scale data
        data_scaled = scaler.transform(data)

        # Predict
        model = models[model_type]
        prediction = model.predict(data_scaled)

        model_names = {
            'linear_regression': 'Linear Regression',
            'ann': 'Artificial Neural Network (MLP)',
            'random_forest': 'Random Forest'
        }

        return render_template(
            "index.html",
            prediction_text=f'Prediksi Harga Berlian ({model_names[model_type]}): ${prediction[0]:,.2f}'
        )
    except Exception as e:
        return render_template(
            "index.html",
            error_text=f"Terjadi kesalahan saat memproses input: {str(e)}"
        )

if __name__ == "__main__":
    app.run(debug=True)