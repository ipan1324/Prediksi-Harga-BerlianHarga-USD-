from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Models directory
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Debug Railway
print("BASE_DIR:", BASE_DIR)
print("MODEL_DIR:", MODEL_DIR)

# Check files inside models folder
if os.path.exists(MODEL_DIR):
    print("MODEL FILES:", os.listdir(MODEL_DIR))
else:
    print("MODELS FOLDER NOT FOUND!")

# =========================
# LOAD SCALER
# =========================
scaler = joblib.load(
    os.path.join(MODEL_DIR, "scaler.pkl")
)

# =========================
# LOAD MODELS
# =========================
models = {
    'linear_regression': joblib.load(
        os.path.join(MODEL_DIR, "linear_regression_model.pkl")
    ),

    'ann': joblib.load(
        os.path.join(MODEL_DIR, "ann_model.pkl")
    )

    # Random Forest dihapus karena terlalu besar
}

# =========================
# MODEL METRICS
# =========================
model_metrics = {
    'linear_regression': {
        'r2': 0.8851,
        'mae': 802.45,
        'mse': 1827401.32,
        'confidence': 'Low (±15%)'
    },

    'ann': {
        'r2': 0.9771,
        'mae': 362.18,
        'mse': 365281.90,
        'confidence': 'Medium (±8%)'
    }
}

# =========================
# HOME
# =========================
@app.route('/')
def home():
    return render_template("index.html")

# =========================
# PREDICT
# =========================
@app.route('/predict', methods=['POST'])
def predict():
    try:

        # Check AJAX / JSON request
        is_json = (
            request.is_json or
            request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        )

        form_data = request.get_json() if request.is_json else request.form

        model_type = form_data.get('model_type', 'ann')

        # =========================
        # INPUT DATA
        # =========================
        carat = float(form_data.get('carat'))
        cut = float(form_data.get('cut'))
        color = float(form_data.get('color'))
        clarity = float(form_data.get('clarity'))
        depth = float(form_data.get('depth'))
        table = float(form_data.get('table'))
        x = float(form_data.get('x'))
        y = float(form_data.get('y'))
        z = float(form_data.get('z'))

        # =========================
        # DATAFRAME
        # =========================
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
        ]], columns=[
            'carat',
            'cut',
            'color',
            'clarity',
            'depth',
            'table',
            'x',
            'y',
            'z'
        ])

        # =========================
        # SCALE DATA
        # =========================
        data_scaled = scaler.transform(data)

        # =========================
        # PREDICTIONS
        # =========================
        all_predictions = {}

        for model_name, model in models.items():

            prediction = model.predict(data_scaled)[0]

            # Prevent negative values
            all_predictions[model_name] = max(
                0.0,
                float(prediction)
            )

        selected_prediction = all_predictions[model_type]

        # =========================
        # ERROR MARGINS
        # =========================
        error_margins = {
            'linear_regression': 0.15,
            'ann': 0.08
        }

        margin = error_margins.get(model_type, 0.10)

        lower_bound = max(
            0.0,
            selected_prediction * (1 - margin)
        )

        upper_bound = (
            selected_prediction * (1 + margin)
        )

        # =========================
        # MODEL NAMES
        # =========================
        model_names = {
            'linear_regression': 'Linear Regression',
            'ann': 'Artificial Neural Network (MLP)'
        }

        response_data = {
            'success': True,
            'prediction': selected_prediction,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'all_predictions': all_predictions,
            'selected_model': model_type,
            'selected_model_name': model_names[model_type],
            'confidence_level': model_metrics[model_type]['confidence']
        }

        # JSON response
        if is_json:
            return jsonify(response_data)

        # HTML response
        return render_template(
            "index.html",
            prediction_text=(
                f'Prediksi Harga Berlian '
                f'({model_names[model_type]}): '
                f'{selected_prediction:,.2f} USD'
            ),
            all_predictions=all_predictions,
            selected_model=model_type
        )

    except Exception as e:

        # JSON error response
        if (
            request.is_json or
            request.headers.get('X-Requested-With') == 'XMLHttpRequest'
        ):
            return jsonify({
                'success': False,
                'error': str(e)
            }), 400

        # HTML error response
        return render_template(
            "index.html",
            error_text=f"Terjadi kesalahan: {str(e)}"
        )

# =========================
# METRICS API
# =========================
@app.route('/metrics')
def get_metrics():
    return jsonify(model_metrics)

# =========================
# MAIN
# =========================
if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )