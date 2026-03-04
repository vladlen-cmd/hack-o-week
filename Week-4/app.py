from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
from model import CafeteriaLoadPredictor

app = Flask(__name__, static_folder='static')
CORS(app)

predictor = CafeteriaLoadPredictor()


def initialize():
    """Load data and model on startup."""
    try:
        if os.path.exists('model.pkl') and os.path.exists('cafeteria_data.csv'):
            predictor.load_data()
            predictor.load_model()
            print("✓ Model and data loaded")
        else:
            print("⚠ Model or data not found. Run data_generator.py and model.py first.")
    except Exception as e:
        print(f"⚠ Error during initialization: {e}")


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')


@app.route('/api/historical', methods=['GET'])
def get_historical():
    try:
        if predictor.data is None:
            predictor.load_data()
        
        data = predictor.data.copy()
        data['timestamp'] = data['timestamp'].astype(str)
        
        return jsonify({
            'success': True,
            'data': data.to_dict(orient='records')
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/predict', methods=['GET'])
def get_prediction():
    try:
        if predictor.model is None:
            return jsonify({'success': False, 'error': 'Model not trained.'}), 400
        
        prediction = predictor.predict_next()
        latest = predictor.data.iloc[-1]
        
        meal_names = {0: 'Off-hours', 1: 'Breakfast', 2: 'Lunch', 3: 'Snacks', 4: 'Dinner'}
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'current': {
                'footfall': int(latest['footfall']),
                'electricity': float(latest['electricity_kwh']),
                'meal_period': meal_names.get(int(latest['meal_period']), 'Unknown'),
                'temperature': float(latest['temperature']),
                'is_weekend': bool(latest['is_weekend']),
                'timestamp': str(latest['timestamp'])
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    try:
        if predictor.model is None:
            return jsonify({'success': False, 'error': 'Model not trained.'}), 400
        
        return jsonify({
            'success': True,
            'metrics': predictor.metrics,
            'feature_importance': predictor.get_feature_importance()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/meal-analysis', methods=['GET'])
def get_meal_analysis():
    try:
        if predictor.data is None:
            predictor.load_data()
        
        return jsonify({
            'success': True,
            'meal_analysis': predictor.get_meal_analysis()
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Cafeteria Load Prediction — Dashboard")
    print("=" * 60)
    
    initialize()
    
    print("\n✓ Starting Flask server...")
    print("✓ Dashboard: http://localhost:5003")
    print("✓ API endpoints:")
    print("  - GET  /api/historical")
    print("  - GET  /api/predict")
    print("  - GET  /api/stats")
    print("  - GET  /api/meal-analysis")
    print()
    
    app.run(debug=False, port=5003)
