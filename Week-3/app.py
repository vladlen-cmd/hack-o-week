from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import os
from model import LibraryEnergyPredictor

app = Flask(__name__, static_folder='static')
CORS(app)

predictor = LibraryEnergyPredictor()


def initialize():
    """Load data and model on startup."""
    try:
        if os.path.exists('model.pkl') and os.path.exists('library_data.csv'):
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
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'current': {
                'occupancy': int(latest['occupancy']),
                'electricity': float(latest['electricity_kwh']),
                'is_exam': bool(latest['is_exam_period']),
                'temperature': float(latest['temperature_outside']),
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


@app.route('/api/exam-comparison', methods=['GET'])
def get_exam_comparison():
    try:
        if predictor.data is None:
            predictor.load_data()
        
        comparison = predictor.get_exam_comparison()
        
        return jsonify({
            'success': True,
            'comparison': comparison
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("Library Energy During Exams — Dashboard")
    print("=" * 60)
    
    initialize()
    
    print("\n✓ Starting Flask server...")
    print("✓ Dashboard: http://localhost:5002")
    print("✓ API endpoints:")
    print("  - GET  /api/historical")
    print("  - GET  /api/predict")
    print("  - GET  /api/stats")
    print("  - GET  /api/exam-comparison")
    print()
    
    app.run(debug=False, port=5002)
