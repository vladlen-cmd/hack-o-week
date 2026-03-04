from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
from model import ParkingLightingPredictor
app=Flask(__name__,static_folder='static'); CORS(app); pred=ParkingLightingPredictor()
def init():
    try:
        if os.path.exists('model.pkl') and os.path.exists('parking_data.csv'): pred.load_data();pred.load_model()
    except Exception as e: print(f"⚠ {e}")
@app.route('/')
def index(): return send_from_directory('static','index.html')
@app.route('/api/historical')
def hist():
    try:
        if pred.data is None: pred.load_data()
        d=pred.data.copy();d['timestamp']=d['timestamp'].astype(str);return jsonify({'success':True,'data':d.to_dict(orient='records')})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500
@app.route('/api/predict')
def predict():
    try:
        if pred.model is None: return jsonify({'success':False,'error':'Not trained'}),400
        p=pred.predict_next();l=pred.data.iloc[-1]
        return jsonify({'success':True,'prediction':p,'current':{'vehicles':int(l['vehicles']),'electricity':float(l['electricity_kwh']),'is_dark':bool(l['is_dark']),'has_event':bool(l['has_event']),'timestamp':str(l['timestamp'])}})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500
@app.route('/api/stats')
def stats():
    try:
        if pred.model is None: return jsonify({'success':False,'error':'Not trained'}),400
        return jsonify({'success':True,'metrics':pred.metrics,'feature_importance':pred.get_feature_importance()})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500
@app.route('/api/lighting-analysis')
def lighting():
    try: return jsonify({'success':True,'analysis':pred.get_lighting_analysis()})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500
if __name__=='__main__':
    print("="*60);print("Parking Lot Lighting Forecast — Dashboard");print("="*60)
    init();print("\n✓ http://localhost:5007");app.run(debug=False,port=5007)
