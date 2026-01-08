from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
from arima_model import ClassroomElectricityForecaster

app = Flask(__name__, static_folder='static')
CORS(app)

forecaster = ClassroomElectricityForecaster()

def initialize_model():
    try:
        if os.path.exists('arima_model.pkl') and os.path.exists('classroom_data.csv'):
            forecaster.load_data()
            forecaster.load_model()
            print("✓ Model loaded from disk")
        else:
            print("⚠ Model not found, will need to train first")
    except Exception as e:
        print(f"⚠ Error loading model: {e}")

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/api/historical', methods=['GET'])
def get_historical_data():
    try:
        if forecaster.data is None:
            forecaster.load_data()
        
        data = forecaster.data.reset_index()
        data['timestamp'] = data['timestamp'].astype(str)
        
        return jsonify({
            'success': True,
            'data': data.to_dict(orient='records')
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/predict', methods=['GET'])
def get_prediction():
    try:
        if forecaster.model_fit is None:
            return jsonify({
                'success': False,
                'error': 'Model not trained. Please train the model first.'
            }), 400
        
        prediction = forecaster.predict_next_hour()
        
        latest = forecaster.data.iloc[-1]
        
        return jsonify({
            'success': True,
            'prediction': {
                'value': round(prediction['prediction'], 2),
                'lower_ci': round(prediction['lower_ci'], 2),
                'upper_ci': round(prediction['upper_ci'], 2),
                'confidence_level': prediction['confidence_level']
            },
            'current': {
                'occupancy': int(latest['occupancy']),
                'electricity': float(latest['electricity_kwh']),
                'timestamp': str(latest.name)
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/retrain', methods=['POST'])
def retrain_model():
    try:
        forecaster.load_data()
        forecaster.train()
        forecaster.save_model()
        
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully'
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        if forecaster.model_fit is None:
            return jsonify({
                'success': False,
                'error': 'Model not trained'
            }), 400
        
        metrics = forecaster.evaluate()
        
        return jsonify({
            'success': True,
            'metrics': metrics
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Classroom Usage Forecasting Dashboard")
    print("=" * 60)
    
    initialize_model()
    
    print("\n✓ Starting Flask server...")
    print("✓ Dashboard: http://localhost:5001")
    print("✓ API endpoints:")
    print("  - GET  /api/historical")
    print("  - GET  /api/predict")
    print("  - POST /api/retrain")
    print("  - GET  /api/stats")
    print("\n")
    
    app.run(debug=False, port=5001)
