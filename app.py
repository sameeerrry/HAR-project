from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
from utils.preprocess import preprocess_input
import sklearn
import random

# Version check (strict)
assert sklearn.__version__ == '1.6.1', f"Scikit-learn version mismatch: {sklearn.__version__}"

app = Flask(__name__)

# Load classical ML models
classical_models = {
    'decision_tree': joblib.load('C:/Users/shukl/Downloads/HAR-project/HAR-project/models/decision_tree_model.pkl'),
    'linear_svc': joblib.load('C:/Users/shukl/Downloads/HAR-project/HAR-project/models/linear_svc_model.pkl'),
    'logistic_regression': joblib.load('C:/Users/shukl/Downloads/HAR-project/HAR-project/models/log_reg_model.pkl'),
    'rbf_svm': joblib.load('C:/Users/shukl/Downloads/HAR-project/HAR-project/models/rbf_svm_model.pkl'),
    'random_forest': joblib.load('C:/Users/shukl/Downloads/HAR-project/HAR-project/models/random_forest_model.pkl')
}

# Load the best LSTM model
lstm_model = load_model('C:/Users/shukl/Downloads/HAR-project/HAR-project/models/lstml3_model.h5')

# Map model output to activity label
label_map = {
    0: "walking",
    1: "walking upstairs",
    2: "walking downstairs",
    3: "sitting",
    4: "standing",
    5: "laying"
}

# Generate sample input

def generate_sample_input():
    return np.round(np.random.uniform(-1.0, 1.0, 128 * 9), 3).tolist()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/sample_input')
def sample_input():
    # Randomly choose between 561 (classical) and 1152 (LSTM) size
    input_size = random.choice([561, 128 * 9])  # Classical or LSTM
    sample = [round(random.uniform(-1.0, 1.0), 4) for _ in range(input_size)]

    return jsonify({"features": sample})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'features' not in data:
        return jsonify({'error': 'Missing "features" in request payload'}), 400

    features = data['features']
    model_name = data.get('model', 'all').strip().lower()

    model_map = {
        'all': 'all_models',
        'all models': 'all_models',
        'decision tree': 'decision_tree',
        'linear svc': 'linear_svc',
        'logistic regression': 'logistic_regression',
        'rbf svm': 'rbf_svm',
        'random forest': 'random_forest',
        'lstm': 'lstm'
    }

    model_choice = model_map.get(model_name)
    if not model_choice:
        return jsonify({'error': f'Unknown model: {model_name}'}), 400

    features_np = np.array(features)

    try:
        # Classical Models
        if model_choice == 'all_models' or model_choice in classical_models:
            try:
                classical_input = features_np.reshape(1, -1)
            except Exception as e:
                classical_input = None
                if model_choice == 'all_models':
                    return jsonify({name: f"Reshape error: {str(e)}" for name in classical_models}), 400
                else:
                    return jsonify(f"Reshape error: {str(e)}"), 400

            if classical_input is not None:
                if model_choice == 'all_models':
                    results = {}
                    for name, model in classical_models.items():
                        try:
                            pred = model.predict(classical_input)[0]
                            label = label_map.get(pred, str(pred))
                            results[name] = label
                        except Exception as e:
                            results[name] = f"Prediction error: {str(e)}"
                    # Also try LSTM if input size matches
                    if features_np.size == 128 * 9:
                        try:
                            lstm_input = features_np.reshape(1, 128, 9)
                            lstm_pred = lstm_model.predict(lstm_input, verbose=0)
                            lstm_label = label_map.get(np.argmax(lstm_pred), str(np.argmax(lstm_pred)))
                            results['lstm'] = lstm_label
                        except Exception as e:
                            results['lstm'] = f"LSTM error: {str(e)}"
                    else:
                        results['lstm'] = "Input size error: LSTM expects 1152 values (128x9)"
                    return jsonify(results)
                else:
                    try:
                        model = classical_models[model_choice]
                        pred = model.predict(classical_input)[0]
                        label = label_map.get(pred, str(pred))
                        return jsonify(label)
                    except Exception as e:
                        return jsonify(f"Prediction error: {str(e)}"), 400

        # LSTM Model
        if model_choice == 'lstm':
            if features_np.size == 128 * 9:
                try:
                    lstm_input = features_np.reshape(1, 128, 9)
                    lstm_pred = lstm_model.predict(lstm_input, verbose=0)
                    label = label_map.get(np.argmax(lstm_pred), str(np.argmax(lstm_pred)))
                    return jsonify(label)
                except Exception as e:
                    return jsonify(f"LSTM error: {str(e)}"), 400
            else:
                return jsonify("Input size error: LSTM expects 1152 values (128x9)"), 400

    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

    # Fallback
    return jsonify({'error': 'Unknown error'}), 500

@app.route('/predict_csv', methods=['POST'])
def predict_csv():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    model_name = request.form.get('model', 'all').strip().lower()

    model_map = {
        'all': 'all_models',
        'all models': 'all_models',
        'decision tree': 'decision_tree',
        'linear svc': 'linear_svc',
        'logistic regression': 'logistic_regression',
        'rbf svm': 'rbf_svm',
        'random forest': 'random_forest',
        'lstm': 'lstm'
    }
    model_choice = model_map.get(model_name)
    if not model_choice:
        return jsonify({'error': f'Unknown model: {model_name}'}), 400

    try:
        df = pd.read_csv(file)
    except Exception as e:
        return jsonify({'error': f'CSV read error: {str(e)}'}), 400

    features = df.values
    results = []

    for row in features:
        row_np = np.array(row)
        try:
            if model_choice == 'all_models':
                row_results = {}
                for name, model in classical_models.items():
                    try:
                        pred = model.predict(row_np.reshape(1, -1))[0]
                        label = label_map.get(pred, str(pred))
                        row_results[name] = label
                    except Exception as e:
                        row_results[name] = f"Prediction error: {str(e)}"
                # Also try LSTM if input size matches
                if row_np.size == 128 * 9:
                    try:
                        lstm_input = row_np.reshape(1, 128, 9)
                        lstm_pred = lstm_model.predict(lstm_input, verbose=0)
                        lstm_label = label_map.get(np.argmax(lstm_pred), str(np.argmax(lstm_pred)))
                        row_results['lstm'] = lstm_label
                    except Exception as e:
                        row_results['lstm'] = f"LSTM error: {str(e)}"
                else:
                    row_results['lstm'] = "Input size error: LSTM expects 1152 values (128x9)"
                results.append(row_results)
            elif model_choice == 'lstm':
                if row_np.size == 128 * 9:
                    try:
                        lstm_input = row_np.reshape(1, 128, 9)
                        lstm_pred = lstm_model.predict(lstm_input, verbose=0)
                        label = label_map.get(np.argmax(lstm_pred), str(np.argmax(lstm_pred)))
                        results.append(label)
                    except Exception as e:
                        results.append(f"LSTM error: {str(e)}")
                else:
                    results.append("Input size error: LSTM expects 1152 values (128x9)")
            else:
                try:
                    model = classical_models[model_choice]
                    pred = model.predict(row_np.reshape(1, -1))[0]
                    label = label_map.get(pred, str(pred))
                    results.append(label)
                except Exception as e:
                    results.append(f"Prediction error: {str(e)}")
        except Exception as e:
            results.append(f"Server error: {str(e)}")

    return jsonify(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
