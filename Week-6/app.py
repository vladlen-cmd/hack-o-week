from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
from model import SportsEnergyPredictor

app = Flask(__name__, static_folder='static')
CORS(app)
predictor = SportsEnergyPredictor()

def initialize():
    try:
        if os.path.exists('model.pkl') and os.path.exists('sports_data.csv'):
            predictor.load_data(); predictor.load_model(); print("✓ Ready")
        else: print("⚠ Run data_generator.py and model.py first.")
    except Exception as e: print(f"⚠ {e}")

@app.route('/')
def index(): return send_from_directory('static','index.html')

@app.route('/api/historical')
def historical():
    try:
        if predictor.data is None: predictor.load_data()
        d=predictor.data.copy(); d['timestamp']=d['timestamp'].astype(str)
        return jsonify({'success':True,'data':d.to_dict(orient='records')})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/predict')
def predict():
    try:
        if predictor.model is None: return jsonify({'success':False,'error':'Not trained'}),400
        p=predictor.predict_next(); l=predictor.data.iloc[-1]
        return jsonify({'success':True,'prediction':p,'current':{'users':int(l['users']),'electricity':float(l['electricity_kwh']),'is_night':bool(l['is_night']),'floodlights':bool(l['floodlights_on']),'timestamp':str(l['timestamp'])}})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/stats')
def stats():
    try:
        if predictor.model is None: return jsonify({'success':False,'error':'Not trained'}),400
        return jsonify({'success':True,'metrics':predictor.metrics,'feature_importance':predictor.get_feature_importance()})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500

@app.route('/api/night-analysis')
def night():
    try:
        return jsonify({'success':True,'analysis':predictor.get_night_analysis()})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500

if __name__=='__main__':
    print("="*60); print("Sports Facility Night Usage — Dashboard"); print("="*60)
    initialize(); print("\n✓ Dashboard: http://localhost:5005")
    app.run(debug=False, port=5005)
