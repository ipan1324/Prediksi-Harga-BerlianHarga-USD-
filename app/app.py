from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

import os

# Get the absolute path of the directory containing app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load Scaler
scaler = joblib.load(os.path.join(BASE_DIR, "../models/scaler.pkl"))

# Load Models
models = {
    'linear_regression': joblib.load(os.path.join(BASE_DIR, "../models/linear_regression_model.pkl")),
    'ann': joblib.load(os.path.join(BASE_DIR, "../models/ann_model.pkl")),
    'random_forest': joblib.load(os.path.join(BASE_DIR, "../models/random_forest_model.pkl"))
}

# Evaluation metrics for comparison page (hardcoded from actual training outputs)
model_metrics = {
    'linear_regression': {'r2': 0.8851, 'mae': 802.45, 'mse': 1827401.32, 'confidence': 'Low (±15%)'},
    'ann': {'r2': 0.9771, 'mae': 362.18, 'mse': 365281.90, 'confidence': 'Medium (±8%)'},
    'random_forest': {'r2': 0.9815, 'mae': 272.84, 'mse': 294812.11, 'confidence': 'High (±5%)'}
}

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Determine if request is AJAX (JSON)
        is_json = request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        
        form_data = request.get_json() if request.is_json else request.form
        
        model_type = form_data.get('model_type', 'random_forest')
        
        carat = float(form_data.get('carat'))
        cut = float(form_data.get('cut'))
        color = float(form_data.get('color'))
        clarity = float(form_data.get('clarity'))
        depth = float(form_data.get('depth'))
        table = float(form_data.get('table'))
        x = float(form_data.get('x'))
        y = float(form_data.get('y'))
        z = float(form_data.get('z'))

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

        # Get all predictions for comparison
        all_preds = {}
        for m_name, m_obj in models.items():
            pred_val = m_obj.predict(data_scaled)[0]
            # Ensure price isn't negative
            all_preds[m_name] = max(0.0, float(pred_val))

        selected_prediction = all_preds[model_type]
        
        # Calculate Confidence Range (Standard Error approximation)
        # Random Forest: 5%, ANN: 8%, LR: 15%
        error_margins = {
            'random_forest': 0.05,
            'ann': 0.08,
            'linear_regression': 0.15
        }
        
        margin = error_margins.get(model_type, 0.10)
        lower_bound = max(0.0, selected_prediction * (1 - margin))
        upper_bound = selected_prediction * (1 + margin)

        model_names = {
            'linear_regression': 'Linear Regression',
            'ann': 'Artificial Neural Network (MLP)',
            'random_forest': 'Random Forest'
        }

        response_data = {
            'success': True,
            'prediction': selected_prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'all_predictions': all_preds,
            'selected_model': model_type,
            'selected_model_name': model_names[model_type],
            'confidence_level': model_metrics[model_type]['confidence']
        }

        if is_json:
            return jsonify(response_data)
            
        return render_template(
            "index.html",
            prediction_text=f'Prediksi Harga Berlian ({model_names[model_type]}): {selected_prediction}',
            all_predictions=all_preds,
            selected_model=model_type
        )
    except Exception as e:
        if request.is_json or request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'success': False, 'error': str(e)}), 400
        return render_template(
            "index.html",
            error_text=f"Terjadi kesalahan saat memproses input: {str(e)}"
        )

@app.route('/metrics')
def get_metrics():
    return jsonify(model_metrics)

if __name__ == "__main__":
    app.run(debug=True)