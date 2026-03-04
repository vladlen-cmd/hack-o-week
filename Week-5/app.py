from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
from model import HVACOptimizer

app = Flask(__name__, static_folder='static')
CORS(app)
predictor = HVACOptimizer()

def initialize():
    try:
        if os.path.exists('model.pkl') and os.path.exists('hvac_data.csv'):
            predictor.load_data(); predictor.load_model()
            print("✓ Model and data loaded")
        else:
            print("⚠ Run data_generator.py and model.py first.")
    except Exception as e:
        print(f"⚠ Error: {e}")

@app.route('/')
def index(): return send_from_directory('static', 'index.html')

@app.route('/api/historical')
def get_historical():
    try:
        if predictor.data is None: predictor.load_data()
        d = predictor.data.copy(); d['timestamp'] = d['timestamp'].astype(str)
        return jsonify({'success': True, 'data': d.to_dict(orient='records')})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/predict')
def get_prediction():
    try:
        if predictor.model is None: return jsonify({'success': False, 'error': 'Model not trained.'}), 400
        pred = predictor.predict_next()
        latest = predictor.data.iloc[-1]
        return jsonify({'success': True, 'prediction': pred, 'current': {
            'occupancy': int(latest['occupancy']), 'hvac': float(latest['hvac_kwh']),
            'equipment_heat': float(latest['equipment_heat_kw']),
            'temp_outside': float(latest['temp_outside']), 'setpoint': float(latest['setpoint']),
            'timestamp': str(latest['timestamp'])}})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats')
def get_stats():
    try:
        if predictor.model is None: return jsonify({'success': False, 'error': 'Model not trained.'}), 400
        return jsonify({'success': True, 'metrics': predictor.metrics,
                        'feature_importance': predictor.get_feature_importance()})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/optimization')
def get_optimization():
    try:
        return jsonify({'success': True, 'insights': predictor.get_optimization_insights()})
    except Exception as e: return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60); print("HVAC Optimization in Labs — Dashboard"); print("=" * 60)
    initialize()
    print("\n✓ Dashboard: http://localhost:5004")
    app.run(debug=False, port=5004)
