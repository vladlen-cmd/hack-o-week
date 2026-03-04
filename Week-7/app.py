from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
import os
from model import AdminBuildingPredictor
app=Flask(__name__,static_folder='static'); CORS(app)
pred=AdminBuildingPredictor()
def init():
    try:
        if os.path.exists('model.pkl') and os.path.exists('admin_data.csv'): pred.load_data();pred.load_model();print("✓ Ready")
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
        return jsonify({'success':True,'prediction':p,'current':{'occupancy':int(l['occupancy']),'electricity':float(l['electricity_kwh']),'is_weekend':bool(l['is_weekend']),'timestamp':str(l['timestamp'])}})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500
@app.route('/api/stats')
def stats():
    try:
        if pred.model is None: return jsonify({'success':False,'error':'Not trained'}),400
        return jsonify({'success':True,'metrics':pred.metrics,'feature_importance':pred.get_feature_importance()})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500
@app.route('/api/weekend-dip')
def dip():
    try: return jsonify({'success':True,'dip':pred.get_weekend_dip()})
    except Exception as e: return jsonify({'success':False,'error':str(e)}),500
if __name__=='__main__':
    print("="*60);print("Admin Building Weekend Dip — Dashboard");print("="*60)
    init();print("\n✓ http://localhost:5006");app.run(debug=False,port=5006)
